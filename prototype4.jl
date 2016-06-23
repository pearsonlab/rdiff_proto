# idea: use a type parameterized on functions as the stack element
using DataStructures
import Base: convert, promote_rule, +, -, *, /, >, <, log

type RDiff <: Real
    val::Float64
    adj::Float64
    RDiff(val, adj) = new(val, adj)
end
typealias Converts Union{Float64, Int, Irrational, Rational}
RDiff(x::Converts) = RDiff(x, 0.)

convert(::Type{RDiff}, x::Converts) = RDiff(x)
promote_rule(::Type{Bool}, ::Type{RDiff}) = RDiff
promote_rule(::Type{Float64}, ::Type{RDiff}) = RDiff
promote_rule{T<:Integer}(::Type{T}, ::Type{RDiff}) = RDiff

value(x::RDiff) = x.val
adjoint(x::RDiff) = x.adj
>(x::RDiff, y::RDiff) = x.val > y.val
<(x::RDiff, y::RDiff) = x.val < y.val

type Node{F<:Function, T1<:Tuple, T2}
    op::F
    inputs::T1
    outputs::T2
end

global varstack = Stack(Node)

function grad(f)

    function ∇f(x)
        y = Array{RDiff}(x)
        f(y)
        for node in varstack
            backprop(node)
        end
        clear!(varstack)
        map(value, y)
    end

    return ∇f
end

function clear!(s::Stack)
    while !isempty(s)
        pop!(s)
    end
    nothing
end

backprop(n::Node) = backprop(n.op, n.inputs..., n.outputs)

macro diff_rule(sig, code)
    fn = _fn_of_vals(sig)
    op = sig.args[1]
    op_args = sig.args[2:end]
    argsyms = Symbol[s.args[1] for s in sig.args[2:end]]
    ex = quote
        function $op($(op_args...))
            res = map(RDiff, $fn)
            n = Node($op, tuple($(argsyms...)), res)
            push!(varstack, n)
            res
        end

        function backprop(::typeof($op), $(op_args...), res)
            $code
            nothing
        end
    end
    esc(ex)
end

_fn_of_vals(x) = x
function _fn_of_vals(ex::Expr)
    if ex.head == :(::)
        out = :(value($(ex.args[1])))
    else
        out = copy(ex)
        for i in eachindex(out.args)
            out.args[i] = _fn_of_vals(out.args[i])
        end
    end
    out
end

@diff_rule log(x::RDiff) begin
    x.adj += res.adj / x.val
end

@diff_rule +(x::RDiff, y::RDiff) begin
    x.adj += res.adj
    y.adj += res.adj
end

@diff_rule -(x::RDiff, y::RDiff) begin
    x.adj += res.adj
    y.adj += -res.adj
end

@diff_rule *(x::RDiff, y::RDiff) begin
    x.adj += res.adj * y.val
    y.adj += res.adj * x.val
end

@diff_rule /(x::RDiff, y::RDiff) begin
    x.adj += res.adj / y.val
    y.adj += -res.adj * x.val / y.val^2
end

@diff_rule -(x::RDiff) begin
    x.adj += -res.adj
end
