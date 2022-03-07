/// @file
#define RAMP_UP_CYCLES 1000

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
  uint         j = 0;                                                           // Neighbour stride index.
  uint         j_min = 0;                                                       // Neighbour stride minimun index.
  uint         j_max = offset[i];                                               // Neighbour stride maximum index.
  uint         k = 0;                                                           // Neighbour tuple index.
  uint         n = central[j_max - 1];                                          // Node index.    
  uint         m = 0;                                                           // Rejection index.      
  uint         r = 0;                                                           // Ramp-up index.

  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// RANDOM GENERATOR //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float        ph_rand           = 0.0f;                                        // Flat random phi.
  float        th_rand           = 0.0f;                                        // Flat random threshold.
  uint4        st_ph             = convert_uint4(state_phi[n]);                 // Random generator state.
  uint4        st_th             = convert_uint4(state_threshold[n]);           // Random generator state.
  float        phi_max           = parameter[0];                                // phi_max.

  // RAMPING UP RANDOM GENERATORS:
  for (r = 0; r < RAMP_UP_CYCLES; r++)
  {
    ph_rand = uint_to_float(xoshiro128pp(&st_ph), 0.0f, phi_max);               // Generating random phi (flat distribution)...
    th_rand = uint_to_float(xoshiro128pp(&st_th), 0.0f, +1.0f);                 // Generating random threshold (flat distribution)...  
    state_phi[n] = convert_int4(st_ph);                                         // Updating random generator state...
    state_threshold[n] = convert_int4(st_th);                                   // Updating random generator state...
  }
}
