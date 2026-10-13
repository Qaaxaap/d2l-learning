{
  description = "d2l 学习仓库的工具链。解释器与科学计算包由 nix profile（~/nix）提供";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system} = {
        default = pkgs.mkShellNoCC {
          # 只装工具。解释器和科学计算包（torch、numpy、pandas、matplotlib、
          # d2l 等）都在 nix profile 里（~/nix 的 modules/packages.nix），
          # 这里再声明一份只会多出一套平行的环境。d2l 是本地打包的，
          # 见 ~/nix/pkgs/d2l.nix。
          packages = with pkgs; [
            uv
            just
            git
          ];

          shellHook = ''
            # uv 不要自己下解释器，用 PATH 上那个（nix profile 的 python3）
            export UV_PYTHON_PREFERENCE=only-system
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

            printf 'd2l devShell | %s | uv %s\n' \
              "$(python3 -V 2>&1)" \
              "$(uv --version 2>/dev/null | cut -d' ' -f2)"
            echo '  just env   看环境详情 | just run <路径>   跑脚本 | just --list 看全部'
          '';
        };

        # 只在本地用的语音提醒环境，开发机没有音频设备。
        # 给 tools/remind.sh 合成中文语音，模型在 ~/.local/share/piper/。
        # piper-tts 默认开 withTrain，会拖进 pytorch-lightning、tensorboard、
        # sudachidict 之类近 800 MiB 的东西，这里全部关掉。
        tts = pkgs.mkShellNoCC {
          packages = [
            (pkgs.piper-tts.override {
              withTrain = false;
              withHTTP = false;
              withAlignment = false;
            })
          ];
        };
      };
    };
}
