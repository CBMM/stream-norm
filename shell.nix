{ nixpkgs ? import <nixpkgs> {} }:

with nixpkgs;
let pythonEnv = pkgs.python27.buildEnv.override {
      extraLibs = with pkgs.python27Packages; [ tensorflow ];
    };
in stdenv.mkDerivation {
    name = "stream-norm-env";
    buildInputs =
      [ pythonEnv
      ];
}
