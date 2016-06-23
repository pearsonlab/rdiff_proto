# idea: use a vector of RDiffs and a stack of Ops
using DataStructures
import Base: convert, promote_rule, +, -, *, /, >, <, log

immutable RDiff <: Real
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

typealias Node Tuple{Function, Vector{Int}, Vector{Int}}

global edges = Vector{RDiff}()
global ops = Stack(Node)

function grad(f)

    function ∇f(x)
        Nx = length(x)
        copy!(edges, x)
        f(y)
        edges[end] = RDiff(value(edges[end]), 1.)
        for op in ops
            backprop(op...)
        end
        map(value, edges[1:Nx])
    end

    return ∇f
end

macro diff_rule(sig, code)
    fn = _fn_of_vals(sig)
    op = sig.args[1]
    argsyms = Symbol[s.args[1] for s in sig.args[2:end]]
    ex = quote
        function $op($(sig.args[2:end]...))
            res = map(RDiff, $fn)
            le = length(edges)
            if ndims(res) == 0
                push!(edges, res)
                push!(ops, ($op, [$(argsyms...)], [le + 1]))
            else
                append!(edges, res[:])
                lr = length(res)
                push!(ops, ($op, [$(argsyms...)], collect(le + (1:lr))))
            end
            res
        end

        function backprop($op, )
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
