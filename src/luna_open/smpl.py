"""SMPL template and training-only Gaussian teacher.

Uses installed smplx APIs, with an externally supplied licensed neutral asset.
Surface sampling and the Gaussian adapter are project-specific implementations.
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from .avatar import Gaussians
from .geometry import matrix_to_quaternion, proper_rotation, quaternion_multiply, transform_points


class SMPLTeacher(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_queries: int = 8192,
        seed: int = 2026,
        pose_blend_shapes: bool = True,
    ):
        super().__init__()
        self.pose_blend_shapes = pose_blend_shapes
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Licensed neutral SMPL asset is required: {model_path}. See docs/assets.md."
            )
        import smplx

        self.body = smplx.SMPL(
            model_path,
            gender="neutral",
            num_betas=10,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        )
        self.body.requires_grad_(False)
        vertices = self.body.v_template
        faces = self.body.faces_tensor
        triangles = vertices[faces]
        area = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        ).norm(dim=-1)
        generator = torch.Generator().manual_seed(seed)
        face_ids = torch.multinomial(area, num_queries, replacement=True, generator=generator)
        uniform = torch.rand(num_queries, 2, generator=generator)
        root = uniform[:, 0].sqrt()
        bary = torch.stack((1 - root, root * (1 - uniform[:, 1]), root * uniform[:, 1]), -1)
        indices = faces[face_ids]
        self.register_buffer("vertex_indices", indices)
        self.register_buffer("barycentric", bary)
        self.register_buffer("anchors", (vertices[indices] * bary[..., None]).sum(-2))
        weights = (self.body.lbs_weights[indices] * bary[..., None]).sum(-2)
        self.register_buffer("weights", weights)
        self.register_buffer("semantic_labels", weights.argmax(-1))

    def interpolate(self, vertices: Tensor) -> Tensor:
        return (vertices[:, self.vertex_indices] * self.barycentric[None, ..., None]).sum(-2)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def shaped_anchors(self, betas: Tensor) -> Tensor:
        from smplx.lbs import blend_shapes

        return self.interpolate(self.body.v_template + blend_shapes(betas, self.body.shapedirs))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def global_motion(self, pose: Tensor, betas: Tensor, body_to_camera: Tensor):
        from smplx.lbs import blend_shapes, vertices2joints

        from .geometry import axis_angle_to_quaternion, quaternion_to_matrix

        shape = self.body.v_template + blend_shapes(betas, self.body.shapedirs)
        root = vertices2joints(self.body.J_regressor, shape)[:, 0]
        rotation = quaternion_to_matrix(axis_angle_to_quaternion(pose[:, :3]))
        pivot_translation = root - torch.einsum("bij,bj->bi", rotation, root)
        camera_rotation = body_to_camera[:, :3, :3] @ rotation
        camera_translation = (
            torch.einsum("bij,bj->bi", body_to_camera[:, :3, :3], pivot_translation)
            + body_to_camera[:, :3, 3]
        )
        return camera_rotation, camera_translation

    @torch.autocast(device_type="cuda", enabled=False)
    def deformation(self, pose: Tensor, betas: Tensor) -> tuple[Tensor, Tensor]:
        from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

        batch = pose.shape[0]
        shape = self.body.v_template + blend_shapes(betas, self.body.shapedirs)
        joints = vertices2joints(self.body.J_regressor, shape)
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch, 24, 3, 3)
        feature = (rotations[:, 1:] - torch.eye(3, device=pose.device)).reshape(batch, -1)
        pose_offsets = (
            (feature @ self.body.posedirs).reshape(batch, -1, 3)
            if self.pose_blend_shapes
            else torch.zeros_like(shape)
        )
        _, joint_transforms = batch_rigid_transform(rotations, joints, self.body.parents)
        per_vertex = torch.einsum("vj,bjmn->bvmn", self.body.lbs_weights, joint_transforms)
        posed_vertices = (
            torch.einsum("bvij,bvj->bvi", per_vertex[..., :3, :3], shape + pose_offsets)
            + per_vertex[..., :3, 3]
        )
        # Surface anchors exactly follow barycentric interpolation of posed mesh.
        # Learned off-surface displacements use interpolated LBS linear parts.
        linear = torch.einsum("kj,bjmn->bkmn", self.weights, joint_transforms)[..., :3, :3]
        return self.interpolate(posed_vertices), linear

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        canonical: Gaussians,
        pose: Tensor,
        betas: Tensor,
        body_to_camera: Tensor,
        detach: bool = True,
    ) -> Gaussians:
        """detach=False trains identity through a fixed annotated teacher.

        detach=True is animator distillation; no teacher/canonical gradient leaks.
        """
        with torch.no_grad():
            surface, linear = self.deformation(pose.float(), betas.float())
            rotation = proper_rotation(body_to_camera[:, None, :3, :3] @ linear)
            rotation_q = matrix_to_quaternion(rotation)
        # Identity means already represent shape. Subtract the corresponding
        # shaped anchor so the annotation's beta deformation is not applied twice.
        displacement = canonical.means.float() - self.shaped_anchors(betas.float())
        posed = surface + torch.einsum("bkij,bkj->bki", linear, displacement)
        means = transform_points(body_to_camera, posed)
        result = Gaussians(
            means,
            quaternion_multiply(rotation_q, canonical.quaternions.float()),
            canonical.scales,
            canonical.opacities,
            canonical.colors,
        )
        return result.detach() if detach else result
