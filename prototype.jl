using DataStructures
import Base: convert, promote_rule, +, -, *, /, >, <, log

type RDiff <: Real
    val::Float64
    adj::Float64
    stack::Nullable{Stack}
    backprop::Function

    RDiff(val, adj, stack, backprop) = new(val, adj, stack, backprop)
    RDiff(val, adj, stack) = new(val, adj, stack)
end
typealias Converts Union{Float64, Int, Irrational, Rational}
RDiff(x::Converts) = RDiff(x, 0., Nullable{Stack}())
RDiff(x::Converts, s::Stack) = RDiff(x, 0., Nullable(s))
RDiff(x::Converts, s::Nullable{Stack}) = RDiff(x, 0., s)

convert(::Type{RDiff}, x::Converts) = RDiff(x)
promote_rule(::Type{Bool}, ::Type{RDiff}) = RDiff
promote_rule(::Type{Float64}, ::Type{RDiff}) = RDiff
promote_rule{T<:Integer}(::Type{T}, ::Type{RDiff}) = RDiff

value(x::RDiff) = x.val
adjoint(x::RDiff) = x.adj
>(x::RDiff, y::RDiff) = x.val > y.val
<(x::RDiff, y::RDiff) = x.val < y.val

function grad(f)
    const varstack = Stack(RDiff)

    function ∇f(x)
        y = map(x -> RDiff(x, 0., varstack), x)
        f(y)
        top(varstack).adj = 1.
        for v in varstack
            v.backprop()
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

#= TODO
- make a function to call backprop on entire Stack
    - should set outcome.adj = 1. and call backprop on every element
- separate stack for every function?
    - would be created in a call to rdiff and closed over
    - RDiff variables need a pointer to this, so that functions know what
      stack to push to
    - can initialize without this, but all inputs need to have it set
- what happens with overwriting?, inplace?, loops?
=#

macro diff_rule(sig, code)
    fn = _fn_of_vals(sig)
    argsyms = Symbol[s.args[1] for s in sig.args[2:end]]

    ex = quote
        function $(sig.args[1])($(sig.args[2:end]...))
            vstack = get_stack($(argsyms...))
            this = RDiff($fn, 0.0, vstack)
            function backprop()
                $code
                nothing
            end
            this.backprop = backprop
            push!(vstack, this)
            this
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

function get_stack(vars...)
    has_stack = filter(x -> !isnull(x.stack), vars)
    get(first(has_stack).stack)
end

@diff_rule log(x::RDiff) begin
    x.adj += this.adj / x.val
end

@diff_rule +(x::RDiff, y::RDiff) begin
    x.adj += this.adj
    y.adj += this.adj
end

@diff_rule -(x::RDiff, y::RDiff) begin
    x.adj += this.adj
    y.adj += -this.adj
end

@diff_rule *(x::RDiff, y::RDiff) begin
    x.adj += this.adj * y.val
    y.adj += this.adj * x.val
end

@diff_rule /(x::RDiff, y::RDiff) begin
    x.adj += this.adj / y.val
    y.adj += -this.adj * x.val / y.val^2
end

@diff_rule -(x::RDiff) begin
    x.adj += -this.adj
end
