/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include "uv_helper.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


 
 // TODO combined with computeCov3D to avoid redundant computation
 // Forward method for creating a view to gaussian coordinate system transformation matrix
 __device__ void computeView2Gaussian(const float3& mean, const glm::vec4 rot, const float* viewmatrix,  float* view2gaussian)
 {
	 // glm matrices use column-major order
	 // Normalize quaternion to get valid rotation
	 glm::vec4 q = rot;// / glm::length(rot);
	 float r = q.x;
	 float x = q.y;
	 float y = q.z;
	 float z = q.w;
 
	 // Compute rotation matrix from quaternion
	 glm::mat3 R = glm::mat3(
		 1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		 2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		 2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	 );
 
	 // transform 3D points in gaussian coordinate system to world coordinate system as follows
	 // new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
	 // so the rots is the gaussian to world transform
 
	 // Gaussian to world transform
	 glm::mat4 G2W = glm::mat4(
		 R[0][0], R[1][0], R[2][0], 0.0f,
		 R[0][1], R[1][1], R[2][1], 0.0f,
		 R[0][2], R[1][2], R[2][2], 0.0f,
		 mean.x, mean.y, mean.z, 1.0f
	 );
 
	 // could be simplied by using pointer
	 // viewmatrix is the world to view transformation matrix
	 glm::mat4 W2V = glm::mat4(
		 viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		 viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		 viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		 viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	 );
 
	 // Gaussian to view transform
	 glm::mat4 G2V = W2V * G2W;
 
	 // inverse of Gaussian to view transform
	 // glm::mat4 V2G_inverse = glm::inverse(G2V);
	 // R = G2V[:, :3, :3]
	 // t = G2V[:, :3, 3]
	 
	 // t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
	 // V2G = torch.zeros((N, 4, 4), device='cuda')
	 // V2G[:, :3, :3] = R.transpose(1, 2)
	 // V2G[:, :3, 3] = t2
	 // V2G[:, 3, 3] = 1.0
	 glm::mat3 R_transpose = glm::mat3(
		 G2V[0][0], G2V[1][0], G2V[2][0],
		 G2V[0][1], G2V[1][1], G2V[2][1],
		 G2V[0][2], G2V[1][2], G2V[2][2]
	 );
 
	 glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	 glm::vec3 t2 = -R_transpose * t;
 
	 view2gaussian[0] = R_transpose[0][0];
	 view2gaussian[1] = R_transpose[0][1];
	 view2gaussian[2] = R_transpose[0][2];
	 view2gaussian[3] = 0.0f;
	 view2gaussian[4] = R_transpose[1][0];
	 view2gaussian[5] = R_transpose[1][1];
	 view2gaussian[6] = R_transpose[1][2];
	 view2gaussian[7] = 0.0f;
	 view2gaussian[8] = R_transpose[2][0];
	 view2gaussian[9] = R_transpose[2][1];
	 view2gaussian[10] = R_transpose[2][2];
	 view2gaussian[11] = 0.0f;
	 view2gaussian[12] = t2.x;
	 view2gaussian[13] = t2.y;
	 view2gaussian[14] = t2.z;
	 view2gaussian[15] = 1.0f;
 }


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* view2gaussians,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	const float* view2gaussian;

	computeView2Gaussian(p_orig, rotations[idx], viewmatrix, view2gaussians + idx * 16);

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	int uv_res,
	float focal_x, float focal_y, 
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ view2gaussian,
	const float* __restrict__ cov3Ds,
	const float* viewmatrix,
	const float3* __restrict__ means3D,
	const float3* __restrict__ scales,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ opacity_residue,
	const float* __restrict__ color_residue,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ is_used)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	float2 ray = { (pixf.x - W/2.) / focal_x, (pixf.y - H/2.) / focal_y };
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16]; // TODO we only need 12
	__shared__ float3 collected_scale[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			for (int ii = 0; ii < 16; ii++)
				collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
			collected_scale[block.thread_rank()] = scales[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float* view2gaussian_j = collected_view2gaussian + j * 16;
			float3 scale_j = collected_scale[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float3 ray_point = { ray.x , ray.y, 1.0 };
			// gof part + ours tex
			float3 cam_pos_local = {view2gaussian_j[12], view2gaussian_j[13], view2gaussian_j[14]};
			float3 ray_local = transformPoint4x3_without_t(ray_point, view2gaussian_j);

			// scale the ray_local and cam_pos_local
			double3 ray_local_scaled = { ray_local.x / scale_j.x, ray_local.y / scale_j.y, ray_local.z / scale_j.z };
			double3 cam_pos_local_scaled = { cam_pos_local.x / scale_j.x, cam_pos_local.y / scale_j.y, cam_pos_local.z / scale_j.z };
			bool add_tex = con_o.w > 0.05f && (opacity_residue != nullptr) && (color_residue != nullptr);
			 
			float this_color_residue[3] = {0.0f, 0.0f, 0.0f};
			float this_opacity_residue = 0.0f;

			if(add_tex){
				double AA = ray_local_scaled.x * ray_local_scaled.x + ray_local_scaled.y * ray_local_scaled.y + ray_local_scaled.z * ray_local_scaled.z;
				double BB = 2 * (ray_local_scaled.x * cam_pos_local_scaled.x + ray_local_scaled.y * cam_pos_local_scaled.y + ray_local_scaled.z * cam_pos_local_scaled.z);
				double CC = cam_pos_local_scaled.x * cam_pos_local_scaled.x + cam_pos_local_scaled.y * cam_pos_local_scaled.y + cam_pos_local_scaled.z * cam_pos_local_scaled.z;
	
				// t is the depth of the gaussian
				float t = -BB/(2*AA);
				double3 intersection_pts = { ray_local_scaled.x * t + cam_pos_local_scaled.x,
					ray_local_scaled.y * t + cam_pos_local_scaled.y,
					ray_local_scaled.z * t + cam_pos_local_scaled.z };
				
				float2 uv_coord_1 = {float(intersection_pts.x), float(intersection_pts.y)};
				float2 uv_coord_2 = {float(intersection_pts.y), float(intersection_pts.z)};
				float2 uv_coord_3 = {float(intersection_pts.z), float(intersection_pts.x)};

				float2 query_uv_1 = square_contract_uv(uv_coord_1, 3.0);
				float2 query_uv_2 = square_contract_uv(uv_coord_2, 3.0);
				float2 query_uv_3 = square_contract_uv(uv_coord_3, 3.0);

				auto query_opacity_residue_1 = [=] __device__ (int u_query, int v_query) -> float {
				return opacity_residue[collected_id[j] * uv_res * uv_res * 3 + u_query * uv_res * 3 + v_query * 3];
				};

				auto query_opacity_residue_2 = [=] __device__ (int u_query, int v_query) -> float {
				return opacity_residue[collected_id[j] * uv_res * uv_res * 3 + u_query * uv_res * 3 + v_query * 3 + 1];
				};

				auto query_opacity_residue_3 = [=] __device__ (int u_query, int v_query) -> float {
				return opacity_residue[collected_id[j] * uv_res * uv_res * 3 + u_query * uv_res * 3 + v_query * 3 + 2];
				};

				auto query_color_residue_1 = [=] __device__ (int u_query, int v_query) -> float3 {
				return make_float3(color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9],
							color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 1],
							color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 2]);
				};

				auto query_color_residue_2 = [=] __device__ (int u_query, int v_query) -> float3 {
				return make_float3(color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 3],
							color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 4],
							color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 5]);
				};

				auto query_color_residue_3 = [=] __device__ (int u_query, int v_query) -> float3 {
				return make_float3(color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 6],
							color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 7],
							color_residue[collected_id[j] * uv_res * uv_res * 9 + u_query * uv_res * 9 + v_query * 9 + 8]);
				};
				// float weight_total = scale_j.x + scale_j.y + scale_j.z;
				// float weight_1 = scale_j.x / weight_total;
				// float weight_2 = scale_j.y / weight_total;
				// float weight_3 = scale_j.z / weight_total;
				// float weight_1 = 1.0f / 3.0f;
				// float weight_2 = 1.0f / 3.0f;
				// float weight_3 = 1.0f / 3.0f;
	
				float weight_total = scale_j.x * scale_j.y + scale_j.y * scale_j.z + scale_j.z * scale_j.x + 1e-6;
				float weight_1 = scale_j.x * scale_j.y / weight_total;
				float weight_2 = scale_j.y * scale_j.z / weight_total;
				float weight_3 = scale_j.z * scale_j.x / weight_total;
				// float2 query_uv = square_contract_uv(uv_coord, 0.5);
				float this_opacity_residue_1 = bilinear_interpolation(query_uv_1, uv_res, query_opacity_residue_1);
				float3 _this_color_residue_1 = bilinear_interpolation_float3(query_uv_1, uv_res, query_color_residue_1);

				float this_opacity_residue_2 = bilinear_interpolation(query_uv_2, uv_res, query_opacity_residue_2);
				float3 _this_color_residue_2 = bilinear_interpolation_float3(query_uv_2, uv_res, query_color_residue_2);

				float this_opacity_residue_3 = bilinear_interpolation(query_uv_3, uv_res, query_opacity_residue_3);
				float3 _this_color_residue_3 = bilinear_interpolation_float3(query_uv_3, uv_res, query_color_residue_3);

				// float this_opacity_residue = (this_opacity_residue_1 + this_opacity_residue_2 + this_opacity_residue_3) / 3.0f;

				// float3 _this_color_residue = {(_this_color_residue_1.x + _this_color_residue_2.x + _this_color_residue_3.x) / 3.0f,
				// 	(_this_color_residue_1.y + _this_color_residue_2.y + _this_color_residue_3.y) / 3.0f,
				// 	(_this_color_residue_1.z + _this_color_residue_2.z + _this_color_residue_3.z) / 3.0f};



				this_opacity_residue = (this_opacity_residue_1 * weight_1 + this_opacity_residue_2 * weight_2 + this_opacity_residue_3 * weight_3);
				
				float3 _this_color_residue = {(_this_color_residue_1.x * weight_1 + _this_color_residue_2.x * weight_2 + _this_color_residue_3.x * weight_3),
					(_this_color_residue_1.y * weight_1 + _this_color_residue_2.y * weight_2 + _this_color_residue_3.y * weight_3),
					(_this_color_residue_1.z * weight_1 + _this_color_residue_2.z * weight_2 + _this_color_residue_3.z * weight_3)};


				// this_opacity_residue = (this_opacity_residue_1 * scale_j.x + this_opacity_residue_2 * scale_j.y + this_opacity_residue_3 * scale_j.z) / (scale_j.x + scale_j.y + scale_j.z);

				// float3 _this_color_residue = {(_this_color_residue_1.x * scale_j.x + _this_color_residue_2.x * scale_j.y + _this_color_residue_3.x * scale_j.z) / (scale_j.x + scale_j.y + scale_j.z),
				// (_this_color_residue_1.y * scale_j.x + _this_color_residue_2.y * scale_j.y + _this_color_residue_3.y * scale_j.z) / (scale_j.x + scale_j.y + scale_j.z),
				// (_this_color_residue_1.z * scale_j.x + _this_color_residue_2.z * scale_j.y + _this_color_residue_3.z * scale_j.z) / (scale_j.x + scale_j.y + scale_j.z)};



				// this_color_residue[3] = {_this_color_residue.x, _this_color_residue.y, _this_color_residue.z};
				this_color_residue[0] = _this_color_residue.x;
				this_color_residue[1] = _this_color_residue.y;
				this_color_residue[2] = _this_color_residue.z;

			}


			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// float alpha = min(0.99f, con_o.w * exp(power));
			if (con_o.w * exp(power) < 1.0f / 255.0f)
				continue;

			float alpha = max(0.00f, min(0.99f, con_o.w * exp(power) + this_opacity_residue));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += clamp(features[collected_id[j] * CHANNELS + ch] + this_color_residue[ch], 0.0f, 100.0f) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			is_used[collected_id[j]] = 1;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	int uv_res,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* view2gaussian,
	const float* cov3Ds,
	const float* viewmatrix,
	const float3* means3D,
	const float3* scales,
	const float* depths,
	const float4* conic_opacity,
	const float* opacity_residue,
	const float* color_residue,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* is_used)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		uv_res,
		focal_x, focal_y,
		means2D,
		colors,
		view2gaussian,
		cov3Ds,
		viewmatrix,
		means3D,
		scales,
		depths,
		conic_opacity,
		opacity_residue,
		color_residue,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		is_used);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* view2gaussians,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		view2gaussians,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}