<!-- # Materials


# Experimental


## Unified Hardening model

::: materials.experimental.uh_model


## Modified Cam Clay (MRM) - experimental
::: materials.experimental.mcc_reg


### Implementation details


Hardening rule
$$
p_c = \left( p_{c0} + p_s \right) \exp \left( \frac{\varepsilon^p_v}{ c_p}  \right)
$$

Elastic law

$$
p = \left( p_{\mathrm{ref}} + p_s \right) \exp \left( \frac{\varepsilon^e_v}{\kappa}  \right)
$$

Bulk modulus
$$
K(p) = \frac{1}{\kappa} (p + p_s)
$$


Yield function


Parameter ps

$$
p_s = \frac{N}{v_c} \exp \left(  \frac{1}{\lambda}  \right)
$$


Parameter Z in terms of CSL intersect

$$
\ln Z = \ln v_c - \lambda \ln \left( \frac{1 + p_s}{p_s} \right)
$$


State boundary layer

Distance between NCL and current state during yielding

$$
\ln v_m = \ln Z
- \lambda \ln \left( \frac{p + p_s}{1 + p_s} \right)
- (\lambda - \kappa) \ln \left( 1+ \frac{m^2}{M^2} \right)
$$ -->
