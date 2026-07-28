pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using Documenter
using Thouless

makedocs(
    sitename="Thouless.jl",
    modules=[Thouless],
    checkdocs=:exports,
    doctest=true,
    format=Documenter.HTML(
        canonical="https://matrixlab-research.github.io/thouless/julia/",
        edit_link="main",
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Model construction" => "model.md",
        "Workflow modules" => "workflows.md",
        "API reference" => "reference.md",
    ],
)
