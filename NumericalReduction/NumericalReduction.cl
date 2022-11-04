__kernel void  NumericalReduction(__global int* data, __local int* localData, __global int* outData)
{
    size_t globalID = get_global_id(0);
    size_t localSize = get_local_size(0);
    size_t localID = get_local_id(0);

    localData[localID] = data[globalID];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = localSize >> 1; i > 0; i >>= 1)
    {
        if(localID < i)
            localData[localID] += localData[localID + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localID == 0)
        outData[get_group_id(0)] = localData[0];
}