# Rdiff_proto

## Important:

Work for this is now in much better shape at: https://github.com/jrevels/ReverseDiffPrototype.jl

## But if you're curious...

This is code used by [@jmxpearson](https://github.com/jmxpearson) in his talk at JuliaCon 2016 to illustrate how reverse mode automatic differentiation could be possible using specialized types.

These implementations are inspired by the [Stan math library](https://github.com/stan-dev/math) as detailed in [this paper](http://arxiv.org/abs/1509.07164). These also require 0.5-dev, since some prototypes rely on each function having its own type.

Roughly, the prototypes in this repo use the following strategies:

1. RDiff types contain two floats (the value and adjoint), a reference to the stack that contains them, and an instance-specific closure to call for reverse mode. Stack is allocated inside grad and closed over.

2. Has an RDiff type that is just a pair of floats and a Node type capturing the operation, its inputs, and outputs, as well as the stack that contains them.

3. The idea here was to have the Node be a typealias for a tuple (operation, inputs, outputs) with separate collections for graph edges and nodes. RDiffs here are immutable.

4. Mutable RDiffs are pairs of floats. Nodes are types parameterized on function, input argument tuple, and output type. Uses a global stack.

5. Like 4, but adds storage for in-place gradient computation.

6. Like 5, but defines special nodes for array operations and in-place updates to backprop arrays.
