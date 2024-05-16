{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    systems.url = "github:nix-systems/default";
    devenv = {
      url = "github:cachix/devenv";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs-python = {
      url = "github:cachix/nixpkgs-python";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    packages = forEachSystem (system: {
      devenv-up = self.devShells.${system}.default.config.procfileScript;
    });

    devShells =
      forEachSystem
      (system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            (
              {
                pkgs,
                config,
                ...
              }: {
                env = {
                  MAMBA_ROOT_PREFIX = "${config.env.DEVENV_STATE}/micromamba";
                  NUMBA_CAPTURED_ERRORS = "new_style";
                  NUMBA_DEBUG_CACHE = "1";
                  PYTHONPATH = config.devenv.root;
                };
                packages = with pkgs; [
                  micromamba
                ];

                enterShell = ''
                  eval "$(micromamba shell hook --shell=bash)"
                  micromamba create -n numbadev -f conda-lock.yml --yes
                  micromamba activate numbadev
                '';

                # https://devenv.sh/reference/options/
                languages.python = {
                  enable = true;
                  version = "3.10";
                };

                # pre-commit.hooks = {
                #   alejandra.enable = true;
                #   flake8.enable = true;
                #   trim-trailing-whitespace.enable = true;
                # };
              }
            )
          ];
        };
      });
  };
}
