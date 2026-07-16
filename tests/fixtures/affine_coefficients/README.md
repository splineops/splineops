# Affine coefficient compatibility corpus

These files are immutable base64 encodings of small NPZ archives produced by
SplineOps 2.0.0. They preserve schema versions 1 and 2 independently of the
currently installed NumPy writer. Tests decode them into a temporary directory
and load them through the public `AffinePlan.load_coefficients` API.

The decoded specimens are pinned by SHA-256 as well as behavior:

- schema 1: `79e3da927b4928fa5555e13258e4c63e672b258f1ca79151dbab02a76be7dd92`
- schema 2: `95b4e5b1a2456fc48bb6521084d35978dd2a054a6533921780851d6807b99728`

Do not regenerate an existing fixture after changing archive code. Add a new
schema fixture and retain the older bytes so backward compatibility remains a
real test rather than a round trip through the current implementation.
