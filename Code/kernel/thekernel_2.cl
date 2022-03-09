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
                        __global int*       m_overflow,                         // Rejection sampling overflow.
                        __global int*       m_overflow_sum,                     // Rejection sampling overflow sum.
                        __global float*     parameter)                          // Parameters.
{ 
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// INDICES ///////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  uint         i = get_global_id(0);                                            // Global index [#].
  uint         j = 0;                                                           // Neighbour stride index.
  uint         j_min = 0;                                                       // Neighbour stride minimun index.
  uint         j_max = offset[i];                                               // Neighbour stride maximum index.
  uint         k = 0;                                                           // Neighbour tuple index.
  uint         n = central[j_max - 1];                                          // Node index.    
  uint         m = 0;                                                           // Rejection index.           
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// CELL VARIABLES //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float4       p                 = position[n];                                 // Central node position.
  float4       c                 = color[n];                                    // Node color.
  float        phi_max           = parameter[0];                                // phi_max parameter.
 
  // COMPUTING STRIDE MINIMUM INDEX:
  if (i == 0)
  {
    j_min = 0;                                                                  // Setting stride minimum (first stride)...
  }
  else
  {
    j_min = offset[i - 1];                                                      // Setting stride minimum (all others)...
  }

  phi[n] = phi_int[n];                                                          // Setting new phi...
  p.z = 0.2f*(phi[n]/phi_max);                                                  // Setting new z position...
  c.xyz = colormap(5.0f*p.z);                                                   // Setting color...
  color[n] = c;                                                                 // Updating color...
  position[n] = p;                                                              // Updating position...
}
