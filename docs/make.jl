using Documenter
using CartesianOperators

makedocs(
    modules = [CartesianOperators],
    authors = "PenguinxCutCell contributors",
    sitename = "CartesianOperators.jl",
    format = Documenter.HTML(
        canonical = "https://PenguinxCutCell.github.io/CartesianOperators.jl",
        repolink = "https://github.com/PenguinxCutCell/CartesianOperators.jl",
        collapselevel = 2,
    ),
    pages = [
        "Home" => "index.md",
        "Operators" => "operators.md",
        "Boundary Conditions" => "boundary-conditions.md",
        "API Reference" => "reference.md",
    ],
    pagesonly = true,
    warnonly = true,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/CartesianOperators.jl",
        push_preview = true,
    )
end
