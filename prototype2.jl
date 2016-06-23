# idea: use a vector of Nodes, where each Node is a operator and arguments
using DataStructures
import Base: convert, promote_rule, +, -, *, /, >, <, log

immutable RDiff <: Real
    val::Float64
    adj::Float64

    RDiff(val, adj) = new(val, adj)
end
typealias Converts Union{Float64, Int, Irrational, Rational}
RDiff(x::Converts) = RDiff(x, 0.)

immutable Node
    inputs::Vector{RDiff}
    outputs::Vector{RDiff}
    op::Function
    container::Vector{Node}
end

convert(::Type{RDiff}, x::Converts) = RDiff(x)
promote_rule(::Type{Bool}, ::Type{RDiff}) = RDiff
promote_rule(::Type{Float64}, ::Type{RDiff}) = RDiff
promote_rule{T<:Integer}(::Type{T}, ::Type{RDiff}) = RDiff

value(x::RDiff) = x.val
adjoint(x::RDiff) = x.adj
>(x::RDiff, y::RDiff) = x.val > y.val
<(x::RDiff, y::RDiff) = x.val < y.val

function grad(f)
    global varstack = Vector{Node}()

    function ∇f(x)
        y = map(x -> RDiff(x, 0., varstack), x)
        f(y)
        varstack[end].outputs[1].adj = 1.
        for n in varstack
            backprop(n)
        end
        map(value, y)
    end

    return ∇f

end

macro diff_rule(sig, code)
    fn = _fn_of_vals(sig)
    argsyms = Symbol[s.args[1] for s in sig.args[2:end]]

    ex = quote
        function $(sig.args[1])($(sig.args[2:end]...))
            vstack = get_container($(argsyms...))
            this = RDiff($fn, 0.0, vstack)
            function backprop()
                $code
                nothing
            end
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

function get_container(vars...)
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
