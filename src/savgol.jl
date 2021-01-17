"""
Polynomial smoothing with the Savitzky Golay filters. Adapted for julia >= 1.0.
Sources: https://github.com/BBN-Q/Qlab.jl/blob/master/src/SavitskyGolay.jl
         https://gist.github.com/lnacquaroli/c97fbc9a15488607e236b3472bcdf097
Requires LinearAlgebra and DSP modules loaded.
"""

# TODO: could probably improve performance with some @inbounds sprinkling
function savitzkyGolay(x::AbstractVector, windowSize::Int, polyOrder::Int; deriv::Int=0)
  
  isodd(windowSize) || throw("Window size must be an odd integer.")
  polyOrder < windowSize || throw("Polynomial order must me less than window size.")
  
  halfWindow = Int( ceil((windowSize-1)/2) )
  
  # Setup the S matrix of basis vectors
  S = zeros.(windowSize, polyOrder+1)
  for ct = 0:polyOrder
    S[:,ct+1] = (-halfWindow:halfWindow).^(ct)
  end
  
  ## Compute the filter coefficients for all orders
  
  # From the scipy code it seems pinv(S) and taking rows should be enough
  G = S * pinv(S' * S)
  
  # Slice out the derivative order we want
  filterCoeffs = G[:, deriv+1] * factorial(deriv)
  
  # Pad the signal with the endpoints and convolve with filter
  paddedX = [x[1]*ones(halfWindow); x; x[end]*ones(halfWindow)]
  y = conv(filterCoeffs[end:-1:1], paddedX)
  
  # Return the valid midsection
  return y[2*halfWindow+1:end-2*halfWindow]
  
end