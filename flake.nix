{
  description = "d2l 学习环境：PyTorch(CUDA) + uv，目标机 RTX 4070 SUPER";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # PyPI 的 CUDA wheel 运行时需要宿主驱动里的这几个库。
      # 只把驱动库单独链接到一个目录，不把整个 /usr/lib 塞进 LD_LIBRARY_PATH，
      # 否则系统 glibc 会盖掉 nix store 里的，nix 程序可能直接崩。
      driverLibs = [
        "libcuda.so.1"
        "libnvidia-ml.so.1"
        "libnvidia-ptxjitcompiler.so.1"
        "libnvidia-nvvm.so.4"
      ];
      driverNames = pkgs.lib.concatStringsSep " " driverLibs;

      # wheel 里 .so 依赖的通用系统库，nix 环境默认不暴露。
      wheelLibs = with pkgs; [ zlib stdenv.cc.cc.lib libGL glib ];
    in
    {
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          python312
          uv
          git
          gnumake
        ];

        shellHook = ''
          D2L_DRV="''${XDG_CACHE_HOME:-$HOME/.cache}/d2l/driver-libs"
          mkdir -p "$D2L_DRV"
          for lib in ${driverNames}; do
            [ -e "$D2L_DRV/$lib" ] && continue
            for cand in /usr/lib/"$lib" /usr/lib64/"$lib" /run/opengl-driver/lib/"$lib"; do
              if [ -e "$cand" ]; then ln -sf "$cand" "$D2L_DRV/$lib"; break; fi
            done
          done
          export LD_LIBRARY_PATH="$D2L_DRV:${pkgs.lib.makeLibraryPath wheelLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          # uv 只用 nix 提供的解释器，不要自己下载一份 python
          export UV_PYTHON_PREFERENCE=only-system
          export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

          echo "d2l devShell: $(python3 -V 2>&1) | uv $(uv --version 2>/dev/null | cut -d' ' -f2)"
          if [ ! -d .venv ]; then
            echo "  尚未创建虚拟环境，先跑: make setup"
          fi
        '';
      };
    };
}
