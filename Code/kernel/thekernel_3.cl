/// @file

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       neighbour,                          // Neighbour.
                        __global int*       offset,                             // Offset. 
                        __global float*     phi,                                // phi.  
                        __global float*     phi_int,                            // phi (intermediate value). 
                        __global int4*      state_phi,                          // Random number generator state.
                        __global int4*      state_threshold,                    // Random number generator state. 
                        __global float*     phi_row_sum,                        // phi row summation.
                        __global float*     phi2_row_sum,                       // phi square row summation.
                        __global float*     parameter)                          // Parameters.
{
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDICES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  uint         i = get_global_id(0);                                            // Global index [#].
  uint         j = 0;                                                           // Row stride index.
  uint         j_min = i*(uint)parameter[10];                                   // Row stride minimum index (based on number of columns).
  uint         j_max = (i + 1)*(uint)parameter[10] - 1;                         // Row stride maximum index (based on number of columns).
  float        phi_partial_sum = 0.0f;                                          // phi partial summation.
  float        phi2_partial_sum = 0.0f;                                         // phi square partial summation.

  // Summating all phi in a row:
  for (j = j_min; j < j_max; j++)
  {
    phi_partial_sum += phi[j];                                                  // Accumulating phi partial summation...
    phi2_partial_sum += pown(phi[j], 2);                                        // Accumulating phi square partial summation...
  }

  phi_row_sum[i] = phi_partial_sum;                                             // Setting phi row summation...
  phi2_row_sum[i] = phi2_partial_sum;                                           // Setting phi square row summation...
}