/// @file

// Central node energy function:
float E_central(float c_1, float c_2, float lambda, float mu, float T, float phi_central)
{
  float E;                                                                      // Energy.
  
  // Computing energy:
  E = (1 - pown(mu, 2) + c_1*pown(T, 2))*pown(phi_central, 2) + 
      (c_2*T)*pown(phi_central, 3) + 
      lambda*pown(phi_central, 4);                         

  return E;
}

// Neighbour node energy function:
float E_neighbour(float C_radial, float phi_neighbour, float phi_central)
{
  float E;                                                                      // Energy.
  
  E = -(C_radial*sin(phi_neighbour)*sin(phi_central));                          // Computing energy...

  return E;
}

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
  float4       neighbour         = (float4)(0.0f, 0.0f, 0.0f, 1.0f);            // Neighbour node position.
  float2       link              = (float2)(0.0f, 0.0f);                        // Neighbour link.
  float        L                 = 0.0f;                                        // Neighbour link length.
  float        dt                = dt_simulation[0];                            // Simulation time step [s].
  float        ds                = ds_simulation[0];                            // Simulation space step.
  uint4        st_phi            = convert_uint4(state_phi[n]);                 // Random generator state.
  uint4        st_threshold      = convert_uint4(state_threshold[n]);           // Random generator state.
  float        c_1               = c_1_parameter[0];                            // c_1 parameter.
  float        c_2               = c_2_parameter[0];                            // c_2 parameter.
  float        lambda            = lambda_parameter[0];                         // lambda parameter.
  float        mu                = mu_parameter[0];                             // mu parameter.
  float        T                 = T_parameter[0];                              // T parameter.
  float        T_hat             = T_hat_parameter[0];                          // T_hat parameter.
  float        phi_max           = phi_max_parameter[0];                        // phi_max parameter.
  float        E                 = 0.0f;                                        // Energy function.
  float        En                = 0.0f;                                        // Energy of central node.
  float        phi_rand          = 0.0f;                                        // Flat random phi.
  float        threshold_rand    = 0.0f;                                        // Flat random threshold.
  float        alpha             = radial_exponent[0];                          // Radial exponent.
  float        D                 = 0.0f;                                        // Distributed random z-spin.
  uint         m_max             = max_rejections[0];                           // Maximum allowed number of rejections.
  float4       c                 = color[n];                                    // Node color.
 
  // COMPUTING STRIDE MINIMUM INDEX:
  if (i == 0)
  {
    j_min = 0;                                                                  // Setting stride minimum (first stride)...
  }
  else
  {
    j_min = offset[i - 1];                                                      // Setting stride minimum (all others)...
  }

  // COMPUTING RANDOM Z-SPIN FROM DISTRIBUTION (rejection sampling):
  do
  {
    phi_rand = uint_to_float(xoshiro128pp(&st_phi), 0.0f, phi_max);             // Generating random theta (flat distribution)...
    threshold_rand = uint_to_float(xoshiro128pp(&st_threshold), 0.0f, +1.0f);   // Generating random threshold (flat distribution)...
    En = E_central(c_1, c_2, lambda, mu, T, phi[n]);                            // Computing central energy term on central theta...
    E = E_central(c_1, c_2, lambda, mu, T, phi_rand);                           // Computing central energy term on random theta...

    // COMPUTING ENERGY:
    for (j = j_min; j < j_max; j++)
    {
      k = nearest[j];                                                           // Computing neighbour index...
      neighbour = position[k];                                                  // Getting neighbour position...
      link = neighbour.xy - p.xy;                                               // Getting neighbour link vector...
      L = length(link);                                                         // Computing neighbour link length...

      if (L == (2.0f + ds))
      {
        L = ds;
      }

      if (L > (2.0f + ds))
      {
        L = sqrt(2.0f)*ds;
      }
      
      En += E_neighbour(0.5f/pow(L/ds, alpha), phi[k], phi[n]);                 // Accumulating neighbour energy terms on central z-spin...
      E += E_neighbour(0.5f/pow(L/ds, alpha), phi[k], phi_rand);                // Accumulating neighbour energy terms on random z-spin...          
    }
    
    D = 1.0f/(1.0f + exp((E - En)/T_hat));                                      // Computing new z-spin candidate from distribution...
    m++;                                                                        // Updating rejection index...
  }
  while ((threshold_rand > D) && (m < m_max));                                  // Evaluating new z-spin candidate (discarding if not found before m_max iterations)...

  phi_int[n] = phi_rand;                                                        // Setting new z-spin (intermediate value)...
  state_phi[n] = convert_int4(st_phi);                                          // Updating random generator state...
  state_threshold[n] = convert_int4(st_threshold);                              // Updating random generator state...
}
