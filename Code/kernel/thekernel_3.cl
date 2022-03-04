/// @file

__kernel void thekernel(__global float4*    color,                              // Color.
                        __global float4*    position,                           // Position.
                        __global int*       central,                            // Node.
                        __global int*       nearest,                            // Neighbour.
                        __global int*       offset,                             // Offset. 
                        __global float*     phi,                                // phi.  
                        __global float*     phi_int,                            // phi (intermediate value). 
                        __global int4*      state_phi,                          // Random number generator state.
                        __global int4*      state_threshold,                    // Random number generator state. 
                        __global int*       max_rejections,                     // Maximum allowed number of rejections. 
                        __global float*     c_1_parameter,                      // c_1 parameter.
                        __global float*     c_2_parameter,                      // c_2 parameter.
                        __global float*     lambda_parameter,                   // lambda parameter.
                        __global float*     mu_parameter,                       // mu parameter.
                        __global float*     T_parameter,                        // T parameter.
                        __global float*     T_hat_parameter,                    // T_hat parameter.
                        __global float*     phi_max_parameter,                  // phi_max parameter.
                        __global float*     radial_exponent,                    // Radial exponent.
                        __global int*       columns,                            // Number of columns in mesh.
                        __global float*     spin_z_row_sum,                     // z-spin row summation.
                        __global float*     spin_z2_row_sum,                    // z-spin square row summation.
                        __global float*     ds_simulation,                      // Mesh side.
                        __global float*     dt_simulation)                      // Simulation time step.
{
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDICES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  uint         i = get_global_id(0);                                            // Global index [#].
  uint         j = 0;                                                           // Row stride index.
  uint         j_min = i*columns[0];                                            // Row stride minimun index.
  uint         j_max = (i + 1)*columns[0] - 1;                                  // Row stride maximum index.
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