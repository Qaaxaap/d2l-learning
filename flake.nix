{
  description = "d2l 学习仓库的工具链。解释器与科学计算包复用系统那套（Arch python-pytorch-cuda）";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShellNoCC {
        # 只装工具。不提供 python，免得遮蔽系统解释器、也和系统里那份
        # 编译好的 torch C 扩展对不上。
        packages = with pkgs; [
          uv
          just
          git
        ];

        shellHook = ''
          # uv 不要自己下解释器，用系统的
          export UV_PYTHON_PREFERENCE=only-system
          export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

          printf 'd2l devShell | %s | uv %s\n' \
            "$(python3 -V 2>&1)" \
            "$(uv --version 2>/dev/null | cut -d' ' -f2)"
          echo '  just env   看环境详情 | just run <路径>   跑脚本 | just --list 看全部'
        '';
      };
    };
}
