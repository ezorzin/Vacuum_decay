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
                        __global float*     spin_z_row_sum,                     // z-spin row summation.
                        __global float*     spin_z2_row_sum,                    // z-spin square row summation.
                        __global float*     parameter)                          // Parameters.
{
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDICES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  uint         i = get_global_id(0);                                            // Global index [#].
  uint         j = 0;                                                           // Row stride index.
  uint         j_min = i*(uint)parameter[9];                                    // Row stride minimum index (base on number of columns).
  uint         j_max = (i + 1)*(uint)parameter[9] - 1;                          // Row stride maximum index (base on number of columns).
  float        spin_z_partial_sum = 0.0f;                                       // z_spin partial summation.
  float        spin_z2_partial_sum = 0.0f;                                      // z_spin square partial summation.

  // Summating all z-spin in a row:
  for (j = j_min; j < j_max; j++)
  {
    spin_z_partial_sum += sin(phi[j]);                                          // Accumulating z-spin partial summation...
    spin_z2_partial_sum += pown(sin(phi[j]), 2);                                // Accumulating z-spin square partial summation...
  }

  spin_z_row_sum[i] = spin_z_partial_sum;                                       // Setting z-spin row summation...
  spin_z2_row_sum[i] = spin_z2_partial_sum;                                     // Setting z-spin square row summation...
}