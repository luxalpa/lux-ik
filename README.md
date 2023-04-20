## Full Body IK for games
This is a full body IK system comparable to Unreal Engine's FBIK. **The project is currently a proof of concept.** The system is based on "Jacobian Pseudo Inverse Damped Least Squares", similar to the one in Unreal Engine, although this one seems to be more accurate.
It is not at all optimized for performance yet, and it takes a bit to converge on a solution.

## Currently Supported features
- Glam
- Positional Goals
- Rotational Goals
- A Rotational Goal that only cares about the Y axis rotation (and leaves the others free)
- Spherical Joints (3 DOF rotations)
- Hinge Joints (1 DOF rotation)
- Joint limits (around arbitrary axes)

## Might be supported in the future
- Joint weights
- Translational Joints
- Other Jacobian solvers (like SVD)
- Configurable constants for Threshold and Damping
- Multiple Iterations per solve
- Auto adjust child joints that aren't part of the chain
- Nalgebra
- Docs, Examples
- More complex joint limits (user defined functions)
- Springy weights
- Option to use Transforms instead of Mat4's

## Contributing
If you want to contribute, please open an issue.