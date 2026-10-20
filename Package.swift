// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "tactus",
    platforms: [.macOS(.v14)],
    products: [.library(name: "Tactus", targets: ["Tactus"]), .executable(name: "tactus-cli", targets: ["tactus-cli"])],
    dependencies: [.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.0")],
    targets: [
        .target(name: "Tactus", dependencies: [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift")]),
        .executableTarget(name: "tactus-cli", dependencies: ["Tactus"]),
        .testTarget(name: "TactusTests", dependencies: ["Tactus", .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift")]),
    ],
    swiftLanguageModes: [.v5]
)
