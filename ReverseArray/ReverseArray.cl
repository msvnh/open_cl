__kernel void ReverseArray(__global int* data, __global int* outData)
{
    size_t size = get_global_size(0) - 1;
    size_t index = size - get_global_id(0);
    outData[get_global_id(0)] = data[index];
}