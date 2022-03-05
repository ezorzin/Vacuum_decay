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
float E_node(float C_radial, float phi_node, float phi_central)
{
  float E;                                                                      // Energy.
  
  E = -(C_radial*sin(phi_node)*sin(phi_central));                               // Computing energy...

  return E;
}

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
  uint         j = 0;                                                           // Neighbour stride index.
  uint         j_min = 0;                                                       // Neighbour stride minimun index.
  uint         j_max = offset[i];                                               // Neighbour stride maximum index.
  uint         k = 0;                                                           // Neighbour tuple index.
  uint         n = central[j_max - 1];                                          // Node index.    
  uint         m = 0;                                                           // Rejection index.           
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////// CELL VARIABLES //////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  float4       c                 = color[n];                                    // Node color.
  float4       p                 = position[n];                                 // Central node position.
  uint4        st_phi            = convert_uint4(state_phi[n]);                 // Random generator state.
  uint4        st_threshold      = convert_uint4(state_threshold[n]);           // Random generator state.
  uint         m_max             = (uint)parameter[0];                          // Maximum allowed number of rejections.
  float        c_1               = parameter[1];                                // c_1 parameter.
  float        c_2               = parameter[2];                                // c_2 parameter.
  float        lambda            = parameter[3];                                // lambda parameter.
  float        mu                = parameter[4];                                // mu parameter.
  float        T                 = parameter[5];                                // T parameter.
  float        T_hat             = parameter[6];                                // T_hat parameter.
  float        phi_max           = parameter[7];                                // phi_max parameter.
  float        alpha             = parameter[8];                                // Radial exponent.
  uint         columns           = (uint)parameter[9];                          // Number of columns.
  float        ds                = parameter[10];                               // Simulation space step.
  float        dt                = parameter[11];                               // Simulation time step [s].
  float4       node              = (float4)(0.0f, 0.0f, 0.0f, 1.0f);            // Neighbour node position.
  float2       link              = (float2)(0.0f, 0.0f);                        // Neighbour link.
  float        L                 = 0.0f;                                        // Neighbour link length.
  float        D                 = 0.0f;                                        // Distributed random z-spin.
  float        E                 = 0.0f;                                        // Energy function.
  float        En                = 0.0f;                                        // Energy of central node.
  float        phi_rand          = 0.0f;                                        // Flat random phi.
  float        threshold_rand    = 0.0f;                                        // Flat random threshold.
  
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
      k = neighbour[j];                                                         // Computing neighbour index...
      node = position[k];                                                       // Getting neighbour position...
      link = node.xy - p.xy;                                                    // Getting neighbour link vector...
      L = length(link);                                                         // Computing neighbour link length...

      if (L == (2.0f + ds))
      {
        L = ds;
      }

      if (L > (2.0f + ds))
      {
        L = sqrt(2.0f)*ds;
      }
      
      En += E_node(0.5f/pow(L/ds, alpha), phi[k], phi[n]);                      // Accumulating neighbour energy terms on central z-spin...
      E += E_node(0.5f/pow(L/ds, alpha), phi[k], phi_rand);                     // Accumulating neighbour energy terms on random z-spin...          
    }
    
    D = 1.0f/(1.0f + exp((E - En)/T_hat));                                      // Computing new z-spin candidate from distribution...
    m++;                                                                        // Updating rejection index...
  }
  while ((threshold_rand > D) && (m < m_max));                                  // Evaluating new z-spin candidate (discarding if not found before m_max iterations)...

  phi_int[n] = phi_rand;                                                        // Setting new z-spin (intermediate value)...
  state_phi[n] = convert_int4(st_phi);                                          // Updating random generator state...
  state_threshold[n] = convert_int4(st_threshold);                              // Updating random generator state...
}
