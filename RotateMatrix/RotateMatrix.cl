__kernel void RotateMatrix(__global int* inData, __global int* outData, size_t total)
{
    size_t id1 = (get_global_id(0) * get_global_size(1)) + get_global_id(1);
    size_t id2 = total - id1; 
    outData[id1] = inData[id2];
}