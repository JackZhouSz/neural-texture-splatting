
#ifndef UV_HELPER_H_INCLUDED
#define UV_HELPER_H_INCLUDED


#include "stdio.h"
#include "auxiliary.h"
#include <cuda_runtime.h>



__forceinline__ __device__ float3 transformPoint4x3_without_t(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}


__forceinline__ __device__ float clamp(float x, float min_v, float max_v)
{
	return min(max(x, min_v), max_v);
}

inline __device__ float2 square_contract_uv(const float2 uv0, int length
) {

    float u = uv0.x / length;
    float v = uv0.y / length;

    // u = clamp(u, -1.0f, 1.0f);
    // v = clamp(v, -1.0f, 1.0f);


    return make_float2(u, v);
}

// todo: contract used in mipnerf360

// inline __device__ float bilinear_interpolation(const float2 uv, const int uv_rest, const float * uv_map){
//     // uv: [-1.0, 1.0]


// }


// inline __device__ float bilinear_interpolation_grad(const float2 uv, const int uv_rest, const float * uv_map, float2 & dvalue_duv){
    
// }

// Bilinear interpolation function with uv_map as a lambda function
template <typename F>
inline __device__ float bilinear_interpolation(const float2 uv, const int uv_rest, F uv_map) {

    if (uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f) {
        return 0.0f;
    }

    // Map uv from [-1.0, 1.0] to [0, uv_rest - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_rest - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_rest - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);

    // Calculate the fractional part
    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_rest - 1);
    int y1 = min(y + 1, uv_rest - 1);

    // Get values at the four corners of the square using the lambda function
    float v00 = uv_map(x, y);
    float v01 = uv_map(x1, y);
    float v10 = uv_map(x, y1);
    float v11 = uv_map(x1, y1);

    // Perform bilinear interpolation
    return (1 - fx) * (1 - fy) * v00 +
           fx * (1 - fy) * v01 +
           (1 - fx) * fy * v10 +
           fx * fy * v11;
}

// Bilinear interpolation function for float3 values
template <typename F>
inline __device__ float3 bilinear_interpolation_float3(const float2 uv, const int uv_res, F uv_map) {
    
    if(uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f){
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);

    // Calculate the fractional part
    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float3 values at the four corners of the square using the lambda function
    float3 v00 = uv_map(x, y);
    float3 v01 = uv_map(x1, y);
    float3 v10 = uv_map(x, y1);
    float3 v11 = uv_map(x1, y1);

    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Perform bilinear interpolation for each component of float3
    float3 result;
    // result.x = (1 - fx) * (1 - fy) * v00.x + fx * (1 - fy) * v01.x + (1 - fx) * fy * v10.x + fx * fy * v11.x;
    // result.y = (1 - fx) * (1 - fy) * v00.y + fx * (1 - fy) * v01.y + (1 - fx) * fy * v10.y + fx * fy * v11.y;
    // result.z = (1 - fx) * (1 - fy) * v00.z + fx * (1 - fy) * v01.z + (1 - fx) * fy * v10.z + fx * fy * v11.z;

    result.x = w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x;
    result.y = w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y;
    result.z = w00 * v00.z + w01 * v01.z + w10 * v10.z + w11 * v11.z;

    return result;
}
template <typename F>
inline __device__ void bilinear_interpolation_float3_grad(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    const float* dL_dcolor_residue_val
) {

    if(uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f){
        return;
    }

    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);
    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float3 values at the four corners of the square using the lambda function
    float3 v00 = uv_map(x, y);    // Top-left
    float3 v01 = uv_map(x1, y);   // Top-right
    float3 v10 = uv_map(x, y1);   // Bottom-left
    float3 v11 = uv_map(x1, y1);  // Bottom-right

    // Calculate gradients with respect to each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res * 3;

    // Accumulate gradients for each corner using atomic operations
    for (int ch = 0; ch < 3; ++ch) {
        atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y * 3 + ch]), w00 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y * 3 + ch]), w01 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y1 * 3 + ch]), w10 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y1 * 3 + ch]), w11 * dL_dcolor_residue_val[ch]);
    }

}

template <typename F>
inline __device__ void bilinear_interpolation_float3_grad_full(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    const float* dL_dcolor_residue_val,
    float2& dL_duv  // Output for dL_du and dL_dv
) {
    if(uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f){
        return;
    }

    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);

    x = max(0, min(x, uv_res - 1));
    y = max(0, min(y, uv_res - 1));

    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float3 values at the four corners of the square using the lambda function
    float3 v00 = uv_map(x, y);    // Top-left
    float3 v01 = uv_map(x1, y);   // Top-right
    float3 v10 = uv_map(x, y1);   // Bottom-left
    float3 v11 = uv_map(x1, y1);  // Bottom-right


    // float3 to float[3]

    float v00_arr[3] = {v00.x, v00.y, v00.z};
    float v01_arr[3] = {v01.x, v01.y, v01.z};
    float v10_arr[3] = {v10.x, v10.y, v10.z};
    float v11_arr[3] = {v11.x, v11.y, v11.z};

    // Calculate gradients with respect to each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res * 3;

    // Initialize gradients with respect to x_mapped and y_mapped
    float dL_dx_mapped = 0.0f;
    float dL_dy_mapped = 0.0f;

    // Accumulate gradients for each corner using atomic operations
    for (int ch = 0; ch < 3; ++ch) {
        // Accumulate gradients to the color residue buffer
        atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y * 3 + ch]), w00 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y * 3 + ch]), w01 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y1 * 3 + ch]), w10 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y1 * 3 + ch]), w11 * dL_dcolor_residue_val[ch]);

        // Compute gradient contributions for x_mapped and y_mapped
        dL_dx_mapped += (v01_arr[ch] - v00_arr[ch]) * (1 - fy) * dL_dcolor_residue_val[ch];
        dL_dx_mapped += (v11_arr[ch] - v10_arr[ch]) * fy * dL_dcolor_residue_val[ch];
        dL_dy_mapped += (v10_arr[ch] - v00_arr[ch]) * (1 - fx) * dL_dcolor_residue_val[ch];
        dL_dy_mapped += (v11_arr[ch] - v01_arr[ch]) * fx * dL_dcolor_residue_val[ch];
    }

    // Convert gradients with respect to `x_mapped` and `y_mapped` to `u` and `v`
    float dL_du = dL_dx_mapped * 0.5f * (uv_res - 1);
    float dL_dv = dL_dy_mapped * 0.5f * (uv_res - 1); // check this?

    dL_duv.x += dL_du;
    dL_duv.y += dL_dv;
}


template <typename F>
inline __device__ void bilinear_interpolation_grad(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    const float dL_dcolor_residue_val
) {
    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);
    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float values at the four corners of the square using the lambda function
    float v00 = uv_map(x, y);    // Top-left
    float v01 = uv_map(x1, y);   // Top-right
    float v10 = uv_map(x, y1);   // Bottom-left
    float v11 = uv_map(x1, y1);  // Bottom-right

    // Calculate weights for each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res;

    // Accumulate gradients for each corner using atomic operations
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res + y]), w00 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res + y]), w01 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res + y1]), w10 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res + y1]), w11 * dL_dcolor_residue_val);
}



template <typename F>
inline __device__ void bilinear_interpolation_float3_grad_full_triplane(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    int plane_id,
    const float* dL_dcolor_residue_val,
    float2& dL_duv  // Output for dL_du and dL_dv
) {
    if(uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f){
        return;
    }

    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);

    x = max(0, min(x, uv_res - 1));
    y = max(0, min(y, uv_res - 1));

    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float3 values at the four corners of the square using the lambda function
    float3 v00 = uv_map(x, y);    // Top-left
    float3 v01 = uv_map(x1, y);   // Top-right
    float3 v10 = uv_map(x, y1);   // Bottom-left
    float3 v11 = uv_map(x1, y1);  // Bottom-right


    // float3 to float[3]

    float v00_arr[3] = {v00.x, v00.y, v00.z};
    float v01_arr[3] = {v01.x, v01.y, v01.z};
    float v10_arr[3] = {v10.x, v10.y, v10.z};
    float v11_arr[3] = {v11.x, v11.y, v11.z};

    // Calculate gradients with respect to each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res * 9 + plane_id * 3;

    // Initialize gradients with respect to x_mapped and y_mapped
    float dL_dx_mapped = 0.0f;
    float dL_dy_mapped = 0.0f;

    // Accumulate gradients for each corner using atomic operations
    for (int ch = 0; ch < 3; ++ch) {
        // Accumulate gradients to the color residue buffer
        atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 9 + y * 9 + ch]), w00 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 9 + y * 9 + ch]), w01 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 9 + y1 * 9 + ch]), w10 * dL_dcolor_residue_val[ch]);
        atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 9 + y1 * 9 + ch]), w11 * dL_dcolor_residue_val[ch]);

        // Compute gradient contributions for x_mapped and y_mapped
        dL_dx_mapped += (v01_arr[ch] - v00_arr[ch]) * (1 - fy) * dL_dcolor_residue_val[ch];
        dL_dx_mapped += (v11_arr[ch] - v10_arr[ch]) * fy * dL_dcolor_residue_val[ch];
        dL_dy_mapped += (v10_arr[ch] - v00_arr[ch]) * (1 - fx) * dL_dcolor_residue_val[ch];
        dL_dy_mapped += (v11_arr[ch] - v01_arr[ch]) * fx * dL_dcolor_residue_val[ch];
    }

    // Convert gradients with respect to `x_mapped` and `y_mapped` to `u` and `v`
    float dL_du = dL_dx_mapped * 0.5f * (uv_res - 1);
    float dL_dv = dL_dy_mapped * 0.5f * (uv_res - 1); // check this?

    dL_duv.x += dL_du;
    dL_duv.y += dL_dv;
}


template <typename F>
inline __device__ void bilinear_interpolation_grad_triplane(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    int plane_idx,
    const float dL_dcolor_residue_val
) {
    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);
    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float values at the four corners of the square using the lambda function
    float v00 = uv_map(x, y);    // Top-left
    float v01 = uv_map(x1, y);   // Top-right
    float v10 = uv_map(x, y1);   // Bottom-left
    float v11 = uv_map(x1, y1);  // Bottom-right

    // Calculate weights for each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res * 3 + plane_idx;

    // Accumulate gradients for each corner using atomic operations
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y * 3]), w00 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y * 3]), w01 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3+ y1 * 3]), w10 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res* 3 + y1 * 3]), w11 * dL_dcolor_residue_val);
}


// template <typename F>
// inline __device__ void bilinear_interpolation_grad_full(
//     const float2 uv, 
//     const int uv_res, 
//     F uv_map, 
//     float* dl_dcolor_residue, 
//     int collected_id, 
//     const float dL_dcolor_residue_val,
//     float2& dL_duv  // Output for dL_du and dL_dv
// ) {
//     // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
//     float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
//     float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

//     // Extract integer coordinates (top-left corner of the square)
//     int x = static_cast<int>(x_mapped);
//     int y = static_cast<int>(y_mapped);

//     // Calculate the fractional part
//     float fx = x_mapped - x;
//     float fy = y_mapped - y;

//     // Ensure that coordinates are within bounds
//     int x1 = min(x + 1, uv_res - 1);
//     int y1 = min(y + 1, uv_res - 1);

//     // Get float values at the four corners using the lambda function
//     float v00 = uv_map(x, y);    // Top-left
//     float v01 = uv_map(x1, y);   // Top-right
//     float v10 = uv_map(x, y1);   // Bottom-left
//     float v11 = uv_map(x1, y1);  // Bottom-right

//     // Compute weights for each corner
//     float w00 = (1 - fx) * (1 - fy);
//     float w01 = fx * (1 - fy);
//     float w10 = (1 - fx) * fy;
//     float w11 = fx * fy;

//     // Compute base index for collected_id in the gradient array
//     int base_idx = collected_id * uv_res * uv_res;

//     // Accumulate gradients for each corner using atomic operations
//     atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res + y]), w00 * dL_dcolor_residue_val);
//     atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res + y]), w01 * dL_dcolor_residue_val);
//     atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res + y1]), w10 * dL_dcolor_residue_val);
//     atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res + y1]), w11 * dL_dcolor_residue_val);

//     // Compute partial derivatives of the interpolated value with respect to fx and fy
//     float dv_dfx = (v01 - v00) * (1 - fy) + (v11 - v10) * fy;
//     float dv_dfy = (v10 - v00) * (1 - fx) + (v11 - v01) * fx;

//     // Compute derivatives of fx and fy with respect to uv
//     float dfx_du = 0.5f * (uv_res - 1);
//     float dfy_dv = 0.5f * (uv_res - 1);

//     // Compute gradients with respect to uv coordinates
//     float dL_du = dL_dcolor_residue_val * dv_dfx * dfx_du;
//     float dL_dv = dL_dcolor_residue_val * dv_dfy * dfy_dv;

//     // Output the gradients with respect to uv
//     // dL_duv = make_float2(dL_du, dL_dv);
//     dL_duv.x += dL_du;
//     dL_duv.y += dL_dv;
// }


template <typename F>
inline __device__ void bilinear_interpolation_float_grad_full(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    const float dL_dcolor_residue_val,
    float2& dL_duv  // Output for dL_du and dL_dv
) {
    if (uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f) {
        return;
    }

    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);

    x = max(0, min(x, uv_res - 1));
    y = max(0, min(y, uv_res - 1));

    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float3 values at the four corners of the square using the lambda function
    float v00 = uv_map(x, y);    // Top-left
    float v01 = uv_map(x1, y);   // Top-right
    float v10 = uv_map(x, y1);   // Bottom-left
    float v11 = uv_map(x1, y1);  // Bottom-right



    // Calculate gradients with respect to each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res;

    // Initialize gradients with respect to x_mapped and y_mapped
    float dL_dx_mapped = 0.0f;
    float dL_dy_mapped = 0.0f;

    // atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y * 3 + ch]), w00 * dL_dcolor_residue_val[ch]);
    // atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y * 3 + ch]), w01 * dL_dcolor_residue_val[ch]);
    // atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y1 * 3 + ch]), w10 * dL_dcolor_residue_val[ch]);
    // atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y1 * 3 + ch]), w11 * dL_dcolor_residue_val[ch]);
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res  + y ]), w00 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res + y  ]), w01 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res  + y1  ]), w10 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res  + y1 ]), w11 * dL_dcolor_residue_val);

    // Compute gradient contributions for x_mapped and y_mapped
    // dL_dx_mapped += (v01_arr[ch] - v00_arr[ch]) * (1 - fy) * dL_dcolor_residue_val[ch];
    // dL_dx_mapped += (v11_arr[ch] - v10_arr[ch]) * fy * dL_dcolor_residue_val[ch];
    dL_dx_mapped += (v01 - v00) * (1 - fy) * dL_dcolor_residue_val;
    dL_dx_mapped += (v11 - v10) * fy * dL_dcolor_residue_val;
    // dL_dy_mapped += (v10_arr[ch] - v00_arr[ch]) * (1 - fx) * dL_dcolor_residue_val[ch];
    // dL_dy_mapped += (v11_arr[ch] - v01_arr[ch]) * fx * dL_dcolor_residue_val[ch];
    dL_dy_mapped += (v10 - v00) * (1 - fx) * dL_dcolor_residue_val;
    dL_dy_mapped += (v11 - v01) * fx * dL_dcolor_residue_val;

    // Convert gradients with respect to `x_mapped` and `y_mapped` to `u` and `v`
    float dL_du = dL_dx_mapped * 0.5f * (uv_res - 1);
    float dL_dv = dL_dy_mapped * 0.5f * (uv_res - 1); // check this?

    dL_duv.x += dL_du;
    dL_duv.y += dL_dv;
}

template <typename F>
inline __device__ void bilinear_interpolation_float_grad_full_triplane(
    const float2 uv, 
    const int uv_res, 
    F uv_map, 
    float* dl_dcolor_residue, 
    int collected_id, 
    int plane_idx,
    const float dL_dcolor_residue_val,
    float2& dL_duv  // Output for dL_du and dL_dv
) {
    if (uv.x < -1.0f || uv.x > 1.0f || uv.y < -1.0f || uv.y > 1.0f) {
        return;
    }

    // Map uv from [-1.0, 1.0] to [0, uv_res - 1]
    float x_mapped = ((uv.x + 1.0f) * 0.5f) * (uv_res - 1);
    float y_mapped = ((uv.y + 1.0f) * 0.5f) * (uv_res - 1);

    // Extract integer coordinates (top-left corner of the square)
    int x = static_cast<int>(x_mapped);
    int y = static_cast<int>(y_mapped);

    x = max(0, min(x, uv_res - 1));
    y = max(0, min(y, uv_res - 1));

    float fx = x_mapped - x;
    float fy = y_mapped - y;

    // Ensure that coordinates are within bounds
    int x1 = min(x + 1, uv_res - 1);
    int y1 = min(y + 1, uv_res - 1);

    // Get float3 values at the four corners of the square using the lambda function
    float v00 = uv_map(x, y);    // Top-left
    float v01 = uv_map(x1, y);   // Top-right
    float v10 = uv_map(x, y1);   // Bottom-left
    float v11 = uv_map(x1, y1);  // Bottom-right



    // Calculate gradients with respect to each corner
    float w00 = (1 - fx) * (1 - fy);
    float w01 = fx * (1 - fy);
    float w10 = (1 - fx) * fy;
    float w11 = fx * fy;

    // Calculate base index for collected_id in the gradient array
    int base_idx = collected_id * uv_res * uv_res * 3 + plane_idx;

    // Initialize gradients with respect to x_mapped and y_mapped
    float dL_dx_mapped = 0.0f;
    float dL_dy_mapped = 0.0f;

    // atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y * 3 + ch]), w00 * dL_dcolor_residue_val[ch]);
    // atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y * 3 + ch]), w01 * dL_dcolor_residue_val[ch]);
    // atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y1 * 3 + ch]), w10 * dL_dcolor_residue_val[ch]);
    // atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y1 * 3 + ch]), w11 * dL_dcolor_residue_val[ch]);
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3  + y*3 ]), w00 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y*3  ]), w01 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x * uv_res * 3 + y1*3  ]), w10 * dL_dcolor_residue_val);
    atomicAdd(&(dl_dcolor_residue[base_idx + x1 * uv_res * 3 + y1 *3 ]), w11 * dL_dcolor_residue_val);

    // Compute gradient contributions for x_mapped and y_mapped
    // dL_dx_mapped += (v01_arr[ch] - v00_arr[ch]) * (1 - fy) * dL_dcolor_residue_val[ch];
    // dL_dx_mapped += (v11_arr[ch] - v10_arr[ch]) * fy * dL_dcolor_residue_val[ch];
    dL_dx_mapped += (v01 - v00) * (1 - fy) * dL_dcolor_residue_val;
    dL_dx_mapped += (v11 - v10) * fy * dL_dcolor_residue_val;
    // dL_dy_mapped += (v10_arr[ch] - v00_arr[ch]) * (1 - fx) * dL_dcolor_residue_val[ch];
    // dL_dy_mapped += (v11_arr[ch] - v01_arr[ch]) * fx * dL_dcolor_residue_val[ch];
    dL_dy_mapped += (v10 - v00) * (1 - fx) * dL_dcolor_residue_val;
    dL_dy_mapped += (v11 - v01) * fx * dL_dcolor_residue_val;

    // Convert gradients with respect to `x_mapped` and `y_mapped` to `u` and `v`
    float dL_du = dL_dx_mapped * 0.5f * (uv_res - 1);
    float dL_dv = dL_dy_mapped * 0.5f * (uv_res - 1); // check this?

    dL_duv.x += dL_du;
    dL_duv.y += dL_dv;
}


inline __device__ double2 project_to_uv(double3 intersection_pts, float3 scale_j) {
    // Find minimal axis
    int min_axis = 0;
    double min_scale = scale_j.x;
    if (scale_j.y < min_scale) {
        min_scale = scale_j.y;
        min_axis = 1;
    }
    if (scale_j.z < min_scale) {
        min_scale = scale_j.z;
        min_axis = 2;
    }

    // Drop the minimal axis and create UV coordinates
    double2 uv;
    if (min_axis == 0) { // drop x
        uv = make_double2(intersection_pts.y, intersection_pts.z);
    } else if (min_axis == 1) { // drop y
        uv = make_double2(intersection_pts.x, intersection_pts.z);
    } else { // drop z
        uv = make_double2(intersection_pts.x, intersection_pts.y);
    }

    return uv;
}

#endif
