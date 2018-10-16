#ifndef DVO_CUDA_KERNELS_CUH
#define DVO_CUDA_KERNELS_CUH

__device__
float interpolateOnDevice(const float* d_ImgIntensity, float x, float y, int w, int h)
{
    const char nanType = '0';
    float valCur = nanf(&nanType);

    //bilinear interpolation
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float x1_weight = x - static_cast<float>(x0);
    float y1_weight = y - static_cast<float>(y0);
    float x0_weight = 1.0f - x1_weight;
    float y0_weight = 1.0f - y1_weight;

    if (x0 < 0 || x0 >= w)
        x0_weight = 0.0f;
    if (x1 < 0 || x1 >= w)
        x1_weight = 0.0f;
    if (y0 < 0 || y0 >= h)
        y0_weight = 0.0f;
    if (y1 < 0 || y1 >= h)
        y1_weight = 0.0f;
    float w00 = x0_weight * y0_weight;
    float w10 = x1_weight * y0_weight;
    float w01 = x0_weight * y1_weight;
    float w11 = x1_weight * y1_weight;

    float sumWeights = w00 + w10 + w01 + w11;
    float sum = 0.0f;
    if (w00 > 0.0f)
        sum += d_ImgIntensity[y0*w + x0] * w00;
    if (w01 > 0.0f)
        sum += d_ImgIntensity[y1*w + x0] * w01;
    if (w10 > 0.0f)
        sum += d_ImgIntensity[y0*w + x1] * w10;
    if (w11 > 0.0f)
        sum += d_ImgIntensity[y1*w + x1] * w11;

    if (sumWeights > 0.0f)
        valCur = sum / sumWeights;

    return valCur;
}


__global__
void compute_JtR_Kernel(const float* J, const float* residuals, float* b, int m, int n)
{
    int xi = threadIdx.x + blockIdx.x*blockDim.x;
    int yi = threadIdx.y + blockIdx.y*blockDim.y;

    int ind = xi + yi*n;

    if(xi < n && yi < m)
        atomicAdd(&b[xi], J[ind] * residuals[yi]);
}

__global__
void compute_JTJ_Kernel(const float* J, float* A, const float* weights, int validRows, bool useWeights)
{

    int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
    int ind_y = blockIdx.y * blockDim.y + threadIdx.y;

    int n = 6;
    int m = validRows;

    if (ind_x < n && ind_y < m)
    {
        float tmp = 0.f;

        for (int i = 0; i < m; i++)
            if (useWeights)
                tmp += J[i * n + ind_y] * J[i * n + ind_x] * weights[i];
            else
                tmp += J[i * n + ind_y] * J[i * n + ind_x];

        A[ind_y * n + ind_x] = tmp;
    }
}


__global__
void getSquaredResidualsKernel(const float *d_residuals, float *d_squaredResiduals, float *valids, int n)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx < n)
    {
        d_squaredResiduals[idx] = d_residuals[idx] * d_residuals[idx];
        valids[idx] = (d_residuals[idx] != 0.f) ? 1.f : 0.f;
    }
}


__global__
void computeGradientKernel(const float *d_gray, float *d_gradient, int direction, int w, int h)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // Compute start and end
    int dirX = 1;
    int dirY = 0;
    if (direction == 1)
    {
        dirX = 0;
        dirY = 1;
    }

    int yStart = dirY;
    int yEnd = h - dirY;
    int xStart = dirX;
    int xEnd = w - dirX;

    // Compute gradient
    if ((x >= xStart) && (x < xEnd) && (y >= yStart) && (y < yEnd)) {
        float v0;
        float v1;
        if (direction == 1)
        {
            // y-direction
            v0 = d_gray[(y-1)*w + x];
            v1 = d_gray[(y+1)*w + x];
        }
        else
        {
            // x-direction
            v0 = d_gray[y*w + (x-1)];
            v1 = d_gray[y*w + (x+1)];
        }
        d_gradient[y*w + x] = 0.5f * (v1 - v0);
    }
}

__global__
void computeWeightsKernel_Huber(const float* residuals, float* weights, int n, float k)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;

    if(ind < n){
        float abs_residual = std::abs(residuals[ind]);

        if (abs_residual <= k)
            weights[ind] = 1.0f;
        else
            weights[ind] = k / abs_residual;
    }
}

__global__
void computeWeightsKernel_Constant(const float* residuals, float* weights, int n)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;

    if(ind < n){
            weights[ind] = 1.0f;
    }
}
__global__
void computeWeightsKernel_Tukey(const float* residuals, float* weights, int n, float k)
{
    /*
     * Tukey
     */

    int ind = threadIdx.x + blockDim.x * blockIdx.x;

    if(ind < n){
        float abs_residual = std::abs(residuals[ind]);

        if (abs_residual <= k)
            weights[ind] = (1-((residuals[ind]*residuals[ind])/(k*k)))*(1-((residuals[ind]*residuals[ind])/(k*k)));
        else
            weights[ind] = 0.0f;
    }
}

__global__
void applyWeightsKernel(const float* weights, float* residuals, int n)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;

    if(ind < n){
        residuals[ind] *= weights[ind];
    }
}

__global__
void downsampleGreyKernel(float *outputImg, float *inputImg, int w, int h)
{
    int xImg = threadIdx.x + blockDim.x*blockIdx.x;
    int yImg = threadIdx.y + blockDim.y*blockIdx.y;
    int wDown = w/2;
    int hDown = h/2;

    if ((xImg < wDown) && (yImg < hDown)) {
        float sum = 0.f;
        sum += inputImg[xImg*2   + yImg*2*w]     * .25f; // top left pixel
        sum += inputImg[xImg*2+1 + yImg*2*w]     * .25f; // top right pixel
        sum += inputImg[xImg*2   + (2*yImg+1)*w] * .25f; // bottom left pixel
        sum += inputImg[xImg*2+1 + (2*yImg+1)*w] * .25f; // bottom right pixel
        outputImg[xImg + yImg*wDown] = sum;
    }
}

__global__
void downsampleDepthKernel(float *outputImg, float *inputImg, int w, int h)
{
    int xImg = threadIdx.x + blockDim.x*blockIdx.x;
    int yImg = threadIdx.y + blockDim.y*blockIdx.y;
    int wDown = w/2;
    int hDown = h/2;

    if ((xImg < wDown) && (yImg < hDown)) {
        float d0 = inputImg[xImg*2   + yImg*2*w]; // top left pixel
        float d1 = inputImg[xImg*2+1 + yImg*2*w]; // top right pixel
        float d2 = inputImg[xImg*2   + (2*yImg+1)*w]; // bottom left pixel
        float d3 = inputImg[xImg*2+1 + (2*yImg+1)*w]; // bottom right pixel

        int cnt = 0;
        float sum = 0.0f;
        if (d0 != 0.0f)
        {
            sum += 1.0f / d0;
            ++cnt;
        }
        if (d1 != 0.0f)
        {
            sum += 1.0f / d1;
            ++cnt;
        }
        if (d2 != 0.0f)
        {
            sum += 1.0f / d2;
            ++cnt;
        }
        if (d3 != 0.0f)
        {
            sum += 1.0f / d3;
            ++cnt;
        }

        if (cnt > 0)
        {
            float dInv = sum / float(cnt);
            if (dInv != 0.0f)
                outputImg[xImg + yImg*wDown] = 1.0f / dInv;
        }
    }
}

__global__
void calculateErrorKernel(const float *d_grayRef, const float *d_depthRef,
                          const float *d_grayCur,
                          const float *d_rotMat, const float *d_t,
                          const float *d_K, float* d_residuals, int w, int h) {
    // camera intrinsics
    float fx = d_K[0]; // K(0, 0);
    float fy = d_K[4]; // K(1, 1);
    float cx = d_K[6]; // K(0, 2);
    float cy = d_K[7]; // K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;


    // index
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if((x < w) && (y < h))
    {
        size_t off = y*w + x;
        float residual = 0.0f;

        // project 2d point back into 3d using its depth
        float dRef = d_depthRef[y*w + x];
        if (dRef > 0.0)
        {
            float x0 = (static_cast<float>(x) - cx) * fxInv;
            float y0 = (static_cast<float>(y) - cy) * fyInv;
            float scale = 1.0f;
            //scale = std::sqrt(x0*x0 + y0*y0 + 1.0f);
            dRef = dRef * scale;
            x0 = x0 * dRef;
            y0 = y0 * dRef;

            // transform reference 3d point into current frame
            // reference 3d point
            float pt3Ref[3];
            float pt3Cur[3];

            pt3Ref[0] = x0;
            pt3Ref[1] = y0;
            pt3Ref[2] = dRef;

            // pt3Cur = rotMat * pt3Ref + t;
            pt3Cur[0] = d_rotMat[0]*pt3Ref[0] + d_rotMat[3]*pt3Ref[1] + d_rotMat[6]*pt3Ref[2];
            pt3Cur[1] = d_rotMat[1]*pt3Ref[0] + d_rotMat[4]*pt3Ref[1] + d_rotMat[7]*pt3Ref[2];
            pt3Cur[2] = d_rotMat[2]*pt3Ref[0] + d_rotMat[5]*pt3Ref[1] + d_rotMat[8]*pt3Ref[2];

            pt3Cur[0] += d_t[0];
            pt3Cur[1] += d_t[1];
            pt3Cur[2] += d_t[2];

            if (pt3Cur[2] > 0.0f)
            {
                // project 3d point to 2d
                float pt2CurH[3];
                // pt2CurH = K * pt3Cur;
                pt2CurH[0] = d_K[0]*pt3Cur[0] + d_K[3]*pt3Cur[1] + d_K[6]*pt3Cur[2];
                pt2CurH[1] = d_K[1]*pt3Cur[0] + d_K[4]*pt3Cur[1] + d_K[7]*pt3Cur[2];
                pt2CurH[2] = d_K[2]*pt3Cur[0] + d_K[5]*pt3Cur[1] + d_K[8]*pt3Cur[2];

                float ptZinv = 1.0f / pt2CurH[2];
                float px = pt2CurH[0] * ptZinv;
                float py = pt2CurH[1] * ptZinv;

                // interpolate residual
                float valCur = interpolateOnDevice(d_grayCur, px, py, w, h);
                if (!isnan(valCur))
                {
                    float valRef = d_grayRef[off];
                    float valDiff = valRef - valCur;
                    residual = valDiff;
                }
            }
        }
        d_residuals[off] = residual;
    }
}

__global__
void deriveAnalyticKernel(const float *d_depthRef,
                          const float *d_rotMat, const float *d_t,
                          const float *d_K,
                          const float *d_gradX, const float *d_gradY,
                          float* d_J, int w, int h)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // camera intrinsics
    // camera intrinsics
    float fx = d_K[0]; // K(0, 0);
    float fy = d_K[4]; // K(1, 1);
    float cx = d_K[6]; // K(0, 2);
    float cy = d_K[7]; // K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;

    float residualRowJ[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    if((x < w) && (y < h))
    {
        size_t off = y*(size_t)w + x;

        // project 2d point back into 3d using its depth
        float dRef = d_depthRef[off];
        if (dRef > 0.0f)
        {
            float x0 = (static_cast<float>(x) - cx) * fxInv;
            float y0 = (static_cast<float>(y) - cy) * fyInv;
            float scale = 1.0f;
            //scale = std::sqrt(x0*x0 + y0*y0 + 1.0);
            dRef = dRef * scale;
            x0 = x0 * dRef;
            y0 = y0 * dRef;

            // transform reference 3d point into current frame
            // reference 3d point
            float pt3Ref[3];
            float pt3[3];

            pt3Ref[0] = x0;
            pt3Ref[1] = y0;
            pt3Ref[2] = dRef;

            // pt3Cur = rotMat * pt3Ref + t;
            pt3[0] = d_rotMat[0]*pt3Ref[0] + d_rotMat[3]*pt3Ref[1] + d_rotMat[6]*pt3Ref[2];
            pt3[1] = d_rotMat[1]*pt3Ref[0] + d_rotMat[4]*pt3Ref[1] + d_rotMat[7]*pt3Ref[2];
            pt3[2] = d_rotMat[2]*pt3Ref[0] + d_rotMat[5]*pt3Ref[1] + d_rotMat[8]*pt3Ref[2];

            pt3[0] += d_t[0];
            pt3[1] += d_t[1];
            pt3[2] += d_t[2];

            if (pt3[2] > 0.0f)
            {
                // project 3d point to 2d
                float pt2CurH[3];
                // pt2CurH = K * pt3Cur;
                pt2CurH[0] = d_K[0]*pt3[0] + d_K[3]*pt3[1] + d_K[6]*pt3[2];
                pt2CurH[1] = d_K[1]*pt3[0] + d_K[4]*pt3[1] + d_K[7]*pt3[2];
                pt2CurH[2] = d_K[2]*pt3[0] + d_K[5]*pt3[1] + d_K[8]*pt3[2];

                float ptZinv = 1.0f / pt2CurH[2];
                float px = pt2CurH[0] * ptZinv;
                float py = pt2CurH[1] * ptZinv;

                // compute interpolated image gradient
                float dX = interpolateOnDevice(d_gradX, px, py, w, h);
                float dY = interpolateOnDevice(d_gradY, px, py, w, h);
                if (!isnan(dX) && !isnan(dY))
                {
                    dX = fx * dX;
                    dY = fy * dY;
                    float pt3Zinv = 1.0f / pt3[2];

                    // shorter computation
                    residualRowJ[0] = dX * pt3Zinv;
                    residualRowJ[1] = dY * pt3Zinv;
                    residualRowJ[2] = - (dX * pt3[0] + dY * pt3[1]) * pt3Zinv * pt3Zinv;
                    residualRowJ[3] = - (dX * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv - dY * (1 + (pt3[1] * pt3Zinv) * (pt3[1] * pt3Zinv));
                    residualRowJ[4] = + dX * (1.0 + (pt3[0] * pt3Zinv) * (pt3[0] * pt3Zinv)) + (dY * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv;
                    residualRowJ[5] = (- dX * pt3[1] + dY * pt3[0]) * pt3Zinv;
                }
            }
        }

        // set 1x6 Jacobian row for current residual
        // invert Jacobian according to kerl2012msc.pdf (necessary?)
        for (int j = 0; j < 6; ++j)
            d_J[off*6 + j] = - residualRowJ[j];
    }
}

#endif // DVO_CUDA_KERNELS_CUH
