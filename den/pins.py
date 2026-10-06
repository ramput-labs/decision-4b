"""Revisions and digests come from Kev's suite manifests at KEV_COMMIT."""

from __future__ import annotations

from .catalog import File, Hub, Item, Model, SetName, Url, Use, model_hub, rev, sha256, validate, validate_models

KEV_COMMIT = rev("fe64b1274ea7f80d4095866df90666abb03e9cf6")
KEV_SUITES = "jaredpalmer/kev-suites"
KEV_SUITES_REV = rev("cc4bac803e73112689ec327ffa481c519cbc7a05")
_KEV_RAW = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_COMMIT}/evals"
_KEV_LICENCE = "Apache-2.0 (Kev); per-source terms in the suite manifest"

T, C, D, X, M = Use.TRAIN, Use.CALIBRATION, Use.DEVELOPMENT, Use.TEST, Use.METADATA
P, E = Use.TRAINABLE, Use.EVAL_ONLY


def _base(key: str, repo: str, revision: str, params: str, size: int, arch: str, verify: tuple[File, ...]) -> Model:
    return Model(key, "base", model_hub(repo, revision), verify, size, f"{params}, {arch}")


def _reference(key: str, repo: str, revision: str, base: str, size: int, what: str, verify: tuple[File, ...]) -> Model:
    return Model(key, "reference", model_hub(repo, revision), verify, size, f"Kev 1.0, {what} on {base}", base)


BASES: tuple[Model, ...] = (
    _base("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "0.6B", 1203641805, "dense, 28 layers, hidden 1024", (
        File("model.safetensors", sha256("cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba")),
    )),
    _base("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1", "1.7B", 3452692285, "dense, 28 layers, hidden 2048", (
        File("model.safetensors", sha256("6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841")),
    )),
    _base("qwen3-4b", "Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539", "4B", 8056521492, "dense, 36 layers, hidden 2560", (
        File("model-00001-of-00003.safetensors", sha256("4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241")),
        File("model-00002-of-00003.safetensors", sha256("f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826")),
        File("model-00003-of-00003.safetensors", sha256("c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965")),
    )),
    _base("qwen3-8b", "Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", "8B", 16393044987, "dense, 36 layers, hidden 4096", (
        File("model-00001-of-00005.safetensors", sha256("9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b")),
        File("model-00002-of-00005.safetensors", sha256("9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f")),
        File("model-00003-of-00005.safetensors", sha256("c7dd5a191c8da555def0714550375b3c0117751add408fcdda0c2a79ac0186bc")),
        File("model-00004-of-00005.safetensors", sha256("ad8b708792105133e03fe19e2d56c89c709f677f01285f8b3ede8947d59fd9fe")),
        File("model-00005-of-00005.safetensors", sha256("fbf24915d47ea030bb68ab0b9488f4515a907185baa6dc26837c9c3f2326a550")),
    )),
    _base("qwen3-14b", "Qwen/Qwen3-14B-Base", "0b0bd3732e2c374d483664439ea334928b65f304", "14B", 29548208833, "dense, 40 layers, hidden 5120", (
        File("model-00001-of-00008.safetensors", sha256("fd1c6d2cf56f0a67d4d1491e8fee36cc6e5bfd6a114ab4e61667a63763662b3a")),
        File("model-00002-of-00008.safetensors", sha256("92dbde5d38ecb4fb25b405d1c4d46907481317be6646e8492a22463b48fd09a5")),
        File("model-00003-of-00008.safetensors", sha256("60945918edc6fb2eca950ca2d6d350020f31b805d02b7cac53cd1fac96c1a804")),
        File("model-00004-of-00008.safetensors", sha256("dc3757a85266e177e57719b334052137d881f4d5a586a4a239c5844c00f9fbb5")),
        File("model-00005-of-00008.safetensors", sha256("10d1be9a383fd3371f00916d26484a263c08649bfc59ba94a7afe78ae86285ce")),
        File("model-00006-of-00008.safetensors", sha256("c046e2c218ad550b8a535ce001b258913c5d956e243d01c2a67acc8c4bca8294")),
        File("model-00007-of-00008.safetensors", sha256("27a80e8c324120762187ae76b8cb4983e332b6843645d79137cefe964b19ed2a")),
        File("model-00008-of-00008.safetensors", sha256("73bcc7f129c1884540a8ea079cd5e8b3ec07f4cd66bf1b5a71d3d2c276af160f")),
    )),
    _base("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-Base", "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9", "30B-A3B", 61079782422, "MoE, 128 experts, 48 layers, hidden 2048", (
        File("model-00001-of-00016.safetensors", sha256("7fe481b0c3796bee8d4fa63638f4d6d3d0b1c66339ef63a1aa03a4d784c53e73")),
        File("model-00002-of-00016.safetensors", sha256("3b1e762dd99476a4b7c2d7b331432fe3e32e11e0dd7c90c2822bb84df4881ae2")),
        File("model-00003-of-00016.safetensors", sha256("c66ff62c6aeb11e085e2215c64cfc5031b50fbcadaead02d0687054b9bf523ff")),
        File("model-00004-of-00016.safetensors", sha256("534c0b5e5a215d95bbd77f9a034d5d74c3b4258f4d5d3918e56ed63dec1eec49")),
        File("model-00005-of-00016.safetensors", sha256("8082532f02d2473f51828f4ab5377b2747d5d77b934ba8d93a25d72d4d82078c")),
        File("model-00006-of-00016.safetensors", sha256("9fab34042ea6ee3348994dbb9b582773bfd51c54defca758beee4521cf54f0bd")),
        File("model-00007-of-00016.safetensors", sha256("02f0a1c1e62143483d1d0655afee52e22c03d25b6f24c8f6ed3b9d6cf47fbc1b")),
        File("model-00008-of-00016.safetensors", sha256("b2ccbadc878ec61dc09cec19ae0c4d3ff8f25e9c0e659599a7d26eaa411a8a4d")),
        File("model-00009-of-00016.safetensors", sha256("0b9dc84d14919b4c65fa56e8b6ffd67c8cf431861673034275c26279a8dd525b")),
        File("model-00010-of-00016.safetensors", sha256("70cb610487e592d19eea29eadc8785a9e4e4b88ee65b3dbd7ac50d0c54e61a4f")),
        File("model-00011-of-00016.safetensors", sha256("662f6a0cb5607d3be8d52fac7c1f541769fd8426867cde36c588cbcef9e8bca9")),
        File("model-00012-of-00016.safetensors", sha256("c8ddc8ffa628697a3618f953a99cfdc51a34da97d0194ccf1ad9a5ba6022178b")),
        File("model-00013-of-00016.safetensors", sha256("2ea0a94f86a4eba612753a3159d9d06e307556c95873bde88294cb48f8c14f75")),
        File("model-00014-of-00016.safetensors", sha256("05d8098a9924ca0990db663b934550cb07a6287a6590b13e93b14cf139edc268")),
        File("model-00015-of-00016.safetensors", sha256("002936e6733de8b73ef36c815013cdd53f2c969ffe19ad62cc47ecab69909cff")),
        File("model-00016-of-00016.safetensors", sha256("90a2c863affd5c0a6e4bf14ae7e4ac7ff388b6328c3d49aed6ea18c3139c6536")),
    )),
    _base("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68", "0.8B", 1769913749, "dense, 24 layers, hidden 1024", (
        File("model.safetensors-00001-of-00001.safetensors", sha256("c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c")),
        File("tokenizer.json", sha256("fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927")),
    )),
    _base("qwen3.5-2b", "Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c", "2B", 4571206192, "dense, 24 layers, hidden 2048", (
        File("model.safetensors-00001-of-00001.safetensors", sha256("928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d")),
        File("tokenizer.json", sha256("fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927")),
    )),
    _base("qwen3.5-4b", "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", "4B", 9342824751, "dense, 32 layers, hidden 2560", (
        File("model.safetensors-00001-of-00002.safetensors", sha256("df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712")),
        File("model.safetensors-00002-of-00002.safetensors", sha256("590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef")),
        File("tokenizer.json", sha256("fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927")),
    )),
    _base("qwen3.5-9b", "Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404", "9B", 19329310976, "dense, 32 layers, hidden 4096", (
        File("model.safetensors-00001-of-00004.safetensors", sha256("862bf7bba8a50145d19d0ae463931fae515284024736592a73a336bc4dfa54ee")),
        File("model.safetensors-00002-of-00004.safetensors", sha256("bace8e115e11ca93c22f0352a60d2fb0c76ac6d7d1c2993c143b7ad2b6c8868c")),
        File("model.safetensors-00003-of-00004.safetensors", sha256("63a021ac0011cbfc66166e77103327a8b45dee95832e36551f6b4c3337448959")),
        File("model.safetensors-00004-of-00004.safetensors", sha256("1a643bbed669266917b5058b5d3f660c03233599249ff7d8fd083decfe662ae0")),
        File("tokenizer.json", sha256("fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927")),
    )),
    _base("qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B-Base", "0f0813072d2358973511097385626f21fcb6d422", "35B-A3B", 71926986405, "MoE, 256 experts, 40 layers, hidden 2048", (
        File("model.safetensors-00001-of-00014.safetensors", sha256("0a3ca09e49e5f8468ba7b145f242db2ddff69fd1d0df041f03eceabba6d13f3e")),
        File("model.safetensors-00002-of-00014.safetensors", sha256("da4b9e643edb2d42cd7dd476f42d84393344025e6ac3602d26df26aa1c3ca447")),
        File("model.safetensors-00003-of-00014.safetensors", sha256("9896227bfc2081ee06b027cda00ed85d8a7d5470782cbc3cec1c7c00d90b6ddb")),
        File("model.safetensors-00004-of-00014.safetensors", sha256("270a0e62d35949a8b7bdbbec2dfcce5a9dc8423fe0d2b4fd60917839f9d986f2")),
        File("model.safetensors-00005-of-00014.safetensors", sha256("3306916aecce0b3cd9e12aa00f5a28cdb4c5a1ce3d6723ddb6a9063fb7975a27")),
        File("model.safetensors-00006-of-00014.safetensors", sha256("2eb8f3a0309824813f89c287e47bf806f962ccdd17a98338c7f450505b40d6f5")),
        File("model.safetensors-00007-of-00014.safetensors", sha256("9f11f34db5c9531a131bc8b7e3d16a7704de746e3f23a150c06296b2de22b55d")),
        File("model.safetensors-00008-of-00014.safetensors", sha256("7a04143de019935574ae069607221df7b64c73aaee9e86d0bf6c1fe634fe223b")),
        File("model.safetensors-00009-of-00014.safetensors", sha256("d2c224c67e11866f83356b8c56320cddf7a2c84e29306bf8b4abfe4d97f17968")),
        File("model.safetensors-00010-of-00014.safetensors", sha256("a5315874320efdbd4df5a4b1bf7707468918455016c51f83382993f50ba15751")),
        File("model.safetensors-00011-of-00014.safetensors", sha256("1be21efee407839f1585b6e811f70d3a2d0f01815b4b8ad77f1104b0e524e00e")),
        File("model.safetensors-00012-of-00014.safetensors", sha256("3543da914abbbf17d0c8b97de38f60d988acd42576b4b1d49bce31c8a44e9186")),
        File("model.safetensors-00013-of-00014.safetensors", sha256("6899a73fb1e539089d15dd16c75e36f0208f5a3581d5cc525aa046f5f95a8b86")),
        File("model.safetensors-00014-of-00014.safetensors", sha256("901d0d82c3b4b97850fd7332cabe0e7ad4616485aebc5835a4edfb94a2d9b444")),
        File("tokenizer.json", sha256("fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927")),
    )),
    _base("qwen3.8-27b", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "27B", 55586114863, "dense, 64 layers, hidden 5120", (
        File("model-00001-of-00018.safetensors", sha256("ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c")),
        File("model-00002-of-00018.safetensors", sha256("06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb")),
        File("model-00003-of-00018.safetensors", sha256("2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968")),
        File("model-00004-of-00018.safetensors", sha256("511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a")),
        File("model-00005-of-00018.safetensors", sha256("635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701")),
        File("model-00006-of-00018.safetensors", sha256("0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0")),
        File("model-00007-of-00018.safetensors", sha256("80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892")),
        File("model-00008-of-00018.safetensors", sha256("7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79")),
        File("model-00009-of-00018.safetensors", sha256("af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8")),
        File("model-00010-of-00018.safetensors", sha256("163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9")),
        File("model-00011-of-00018.safetensors", sha256("5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8")),
        File("model-00012-of-00018.safetensors", sha256("a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55")),
        File("model-00013-of-00018.safetensors", sha256("06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a")),
        File("model-00014-of-00018.safetensors", sha256("4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b")),
        File("model-00015-of-00018.safetensors", sha256("69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22")),
        File("model-00016-of-00018.safetensors", sha256("73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe")),
        File("model-00017-of-00018.safetensors", sha256("beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8")),
        File("model-00018-of-00018.safetensors", sha256("1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06")),
        File("tokenizer.json", sha256("0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3")),
    )),
)

REFERENCES: tuple[Model, ...] = (
    _reference("kev-0.8b", "jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e", "qwen3.5-0.8b", 65559936, "LoRA + pointer head", (
        File("adapter_model.safetensors", sha256("9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a")),
        File("head.pt", sha256("f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6")),
    )),
    _reference("kev-4b", "jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101", "qwen3.5-4b", 159741871, "LoRA + pointer head", (
        File("adapter_model.safetensors", sha256("90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2")),
        File("head.pt", sha256("dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c")),
    )),
    _reference("kev-9b", "jaredpalmer/kev-9b", "b5d8c18e44c60888d138b65cb6507ff0a5a448a0", "qwen3.5-9b", 201675340, "LoRA + pointer head", (
        File("adapter_model.safetensors", sha256("2b2a70cf4ef4440b6c22899e1f72c2f8ea5c6f65b19aa344539b4b8971d1f13d")),
        File("head.pt", sha256("8e1dab2c8e3664f6fee843e0257d57946c25e0210c61901ea77e71761f4244e1")),
    )),
    _reference("kev-27b", "jaredpalmer/kev-27b", "28be62e9c5ae0471bda2b7c636a55224b4f4b887", "qwen3.8-27b", 51279880924, "full weights + pointer head", (
        File("head.pt", sha256("7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad")),
        File("model-00001-of-00011.safetensors", sha256("d934f5d5d625a627af59f8f7deca1f915bf165050f5c55834a25080d5cd3f68a")),
        File("model-00002-of-00011.safetensors", sha256("983a355958ed0c1ed8eb4ef756bf8083db445da9b98447194944fba25ca91a76")),
        File("model-00003-of-00011.safetensors", sha256("47cf3227ae3f48b31e53200dab72ac597d79479a0dd0b742fb8e9a41e4e940e3")),
        File("model-00004-of-00011.safetensors", sha256("6ae162fc2aa03f04c8a9ccbcf35bf44a76d410bbc723a6d0225b520a86f02400")),
        File("model-00005-of-00011.safetensors", sha256("0455fa671e71b4861ca7480a25ba6f61718ce7d6bf17b121580e6cc8314c0cb3")),
        File("model-00006-of-00011.safetensors", sha256("91f42758762ef0f62cc18e4d968b7efda7267bbf8ae6818df16767a17bbc4c91")),
        File("model-00007-of-00011.safetensors", sha256("15f979818d2325cd886b022e966e7e15be179e573e8d9f78e29ae0189e304240")),
        File("model-00008-of-00011.safetensors", sha256("e235645cb7aa8e4d31823963cb89d52a11eaad3f4a83768f47ada686246a41bd")),
        File("model-00009-of-00011.safetensors", sha256("a87e281e36d15217a97a670bc58596a683978bd8a117c5d7d14d36cdf5a5c7fe")),
        File("model-00010-of-00011.safetensors", sha256("7a5942495d9188c030ec12a498b3c16ca6d1500d04cbc33317dd55cb8e7d3a0a")),
        File("model-00011-of-00011.safetensors", sha256("60ecc5aa8f4d6db13d312027e640e6a6601d09bc4812ef9fc93859fe6a73830e")),
        File("tokenizer.json", sha256("06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523")),
    )),
)


DEFAULT_MODEL = "qwen3.5-4b"
MODELS: dict[str, Model] = {m.key: m for m in BASES + REFERENCES}
validate_models(BASES + REFERENCES)


# breadth-v1 has no download: `scripts/breadth.py` rebuilds it from the raws below and must match Kev's manifest
BREADTH_SHA256 = {
    "development": "9aad8f4a7374988b2768d31135d5690137df3a3d6cbc1e771db4ee95ca515a94",
    "test": "91a64f0a9448dcaece421c051a4b3fd079c15b84ea6cb9cfa8155237d8d1fa18",
}
BREADTH = {  # what `den audit` checks the built files against
    "dev/breadth.jsonl": {"sha256": BREADTH_SHA256["development"], "records": 1990, "questions": 3075},
    "test/breadth.jsonl": {"sha256": BREADTH_SHA256["test"], "records": 1990, "questions": 3089},
}


_SUITE_FILE: dict[Use, tuple[str, str]] = {T: ("train", ".jsonl"), C: ("calibration", ".jsonl"), D: ("dev", ".jsonl"), X: ("test", ".jsonl"), M: ("manifests", ".json")}


def _suite(suite: str, use: Use, origin: str, digest: str | None, note: str = "") -> Item:
    """origin: "hub:<path in kev-suites>" or "git:<path under evals/ in Kev's repo>"."""
    directory, ext = _SUITE_FILE[use]
    name = f"{suite}{ext}"
    pin = sha256(digest) if digest else None
    where, _, path = origin.partition(":")
    match where:
        case "hub":
            source: Hub | Url = Hub(KEV_SUITES, KEV_SUITES_REV, files=(File(path, pin, name=name),))
        case "git":
            source = Url(f"{_KEV_RAW}/{path}", name, pin)
        case _:
            raise ValueError(f"{suite}: bad origin {origin!r}")
    return Item("suites", f"{suite}/{directory}", use, source, directory, _KEV_LICENCE, note)


SUITES: tuple[Item, ...] = (
    _suite("core", T, "hub:v7/decision-v7/train.jsonl", "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad", "stage 1 (12,576 records); replayed in stages 2-4"),
    _suite("core", C, "hub:v7/decision-v7/calibration.jsonl", "12c2029a07d24828f5f8cc48f09f572c81457de96f3d5d16180be4c090abd5ca"),
    _suite("core", D, "hub:v7/decision-v7/development.jsonl", "8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c", "Kev fit T = 2.41 here"),
    _suite("core", X, "hub:v7/decision-v7/test.jsonl", "cd7d129a84232e0c7a4e92b4840a6dfe14c8bb93c5d1904b526fa8c085325d2d", "locked"),
    _suite("core", M, "hub:v7/decision-v7/manifest.json", None),
    _suite("dates-unknowable", T, "git:night2/dates_unknowable.jsonl", "afd8502d162163605ac446439e32c7b9083302dd78a6bfdc5e30075619e98437", "stage 2 (1,425 records)"),
    _suite("dates-unknowable", M, "git:night2/manifest.json", None),
    _suite("documents", T, "hub:documents-v1/train.jsonl", "2d7e4c4360d642a382d55425d7f34ffa2f46d00ca39a9bae31acaf82b178ee69", "stage 3 (5,219 records)"),
    _suite("documents", D, "git:documents-v1/development.jsonl", "bfca7b41b0d8553b03f9781f48c942dfea368ce19b8508794316e555de8280f5"),
    _suite("documents", X, "git:documents-v1/test.jsonl", "742d04a1abc2207bd3e4f24b281f483cdc69238e4da747cbbc3774e836f2e557"),
    _suite("documents", M, "git:documents-v1/manifest.json", None),
    _suite("skills", T, "hub:hard-v1/train.jsonl", "a08ca9c5ba5d365967876d7d88753cf876624be8553df5af3dca8973e4a1aa4d", "stage 4 (6,000 records)"),
    _suite("skills", D, "git:hard-v1/development.jsonl", "3cdbf12a6b7c3b70c73e8c39f2677127ea61ef32f0ea75b55a43556d884b2d98"),
    _suite("skills", X, "git:hard-v1/test.jsonl", "246ee92234d33f8e3f10ac6e1542651531b128fd7cc66f3a9964589b77dfa22b"),
    _suite("skills", M, "git:hard-v1/manifest.json", None),
    _suite("devtools", T, "git:devtools-v1/train.jsonl", "587682c145cbbc0c8c990e9d649da7b972853a9eaaef71e9b7b0fa18163b962c", "stage 4 (5,320 records)"),
    _suite("devtools", D, "git:devtools-v1/development.jsonl", "3df3aa83890c43db65ac73b5bb6e4fbb1466362fc1a25dbd383d4d2edd0a1fe6"),
    _suite("devtools", X, "git:devtools-v1/test.jsonl", "2fb7c2a239fef923bcb62c0eee9d6884a608ce30e5e74a4e6b9082787679767f"),
    _suite("devtools", M, "git:devtools-v1/manifest.json", None),
    # transfer: out of domain
    _suite("transfer", D, "hub:v4/transfer-v4/development.jsonl", "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e"),
    _suite("transfer", X, "hub:v4/transfer-v4/test.jsonl", "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517", "locked: the headline number"),
    _suite("transfer", M, "hub:v4/transfer-v4/manifest.json", None),
    # probes: dates, MMLU, MMLU-Pro, unknowable
    _suite("probes", D, "hub:v9/transfer-v9/development.jsonl", "53872ba045564315cf3c3dd8fb656e76441c6c514b1a7ff985ff0db3955710cf"),
    _suite("probes", X, "hub:v9/transfer-v9/test.jsonl", "048abea5e12c73c3a622ce5836898c00fa43766e0bf48fb2459e77ee9873497f"),
    _suite("probes", M, "hub:v9/transfer-v9/manifest.json", None),
    _suite("heldout", C, "hub:round3/transfer-r3/calibration.jsonl", None, "held-out sources: the honest place to fit T"),
    _suite("heldout", M, "hub:round3/transfer-r3/manifest.json", None),
    # diagnostics, dev only and never trained on (Kev-4B: binding 0.943, semif 0.847 on its 144 clean rows)
    _suite("binding", D, "git:diagnostics/binding-v1.jsonl", "0b00da23a732db87bddb3c582a02c8d0de9f76cb26d20f8453f73141bbfdacc8", "role binding and date arithmetic in policy cases (560)"),
    _suite("binding", M, "git:diagnostics/binding-v1.manifest.json", None),
    _suite("semif", D, "git:external/semif-v1/development.jsonl", "b9e0c7939a25a04a7ef7bc12334b5fa45b2027d04904d32021295b5539a97303", "SemIf: 144 authored decisions + 108 perturbations (MIT)"),
    _suite("semif", M, "git:external/semif-v1/manifest.json", None),
)


def _hub(set_: SetName, name: str, use: Use, repo: str, revision: str, licence: str, note: str, *, patterns: tuple[str, ...] = (), files: dict[str, str] | None = None) -> Item:
    pinned = tuple(File(path, sha256(digest)) for path, digest in (files or {}).items())
    return Item(set_, name, use, Hub(repo, rev(revision), files=pinned, patterns=patterns), _source_dir(use, name), licence, note)


def _url(set_: SetName, name: str, use: Use, url: str, filename: str, digest: str, licence: str, note: str, *, dest: str | None = None) -> Item:
    return Item(set_, name, use, Url(url, filename, sha256(digest)), _source_dir(use, dest or name), licence, note)


def _source_dir(use: Use, name: str) -> str:
    return f"sources/{'eval' if use is E else 'train'}/{name}"


RAW_TRAIN: tuple[Item, ...] = (
    _hub("raw-train", "intent/banking77", P, "legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec", "CC-BY-4.0", "77 banking intents", patterns=("data/*",)),
    _hub("raw-train", "topic/ag-news", P, "fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "unknown (academic)", "4 news topics", patterns=("data/*",)),
    _hub("raw-train", "topic/trec", P, "CogComp/trec", "65752bf53af25bc935a0dce92fb5b6c930728450", "unknown (academic)", "question type; refs/convert/parquet commit", patterns=("default/*",)),
    _hub("raw-train", "topic/dbpedia14", P, "fancyzhx/dbpedia_14", "9abd46cf7fc8b4c64290f26993c540b92aa145ac", "CC-BY-SA-3.0", "14 ontology classes", patterns=("dbpedia_14/*",)),
    _hub("raw-train", "reading/boolq", P, "google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", "CC-BY-SA-3.0", "yes/no over a passage", patterns=("data/*",)),
    _hub("raw-train", "reading/multinli", P, "nyu-mll/multi_nli", "da70db2af9d09693783c3320c4249840212ee221", "OANC / CC-BY-SA-3.0 mix", "3-way NLI", patterns=("data/*",)),
    _hub("raw-train", "sentiment/sst5", P, "SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3", "unknown (academic)", "5-level sentiment (score)", patterns=("*.jsonl",)),
    _hub("raw-train", "sentiment/yelp-full", P, "Yelp/yelp_review_full", "c1f9ee939b7d05667af864ee1cb066393154bf85", "Yelp dataset terms", "1-5 stars (score)", patterns=("yelp_review_full/*",)),
    _hub("raw-train", "sentiment/amazon-reviews", P, "SetFit/amazon_reviews_multi_en", "ec73b665e4be0f567b69d39425355401cfe0d29b", "Amazon terms (non-commercial research)", "1-5 stars (score)", patterns=("*.jsonl",)),
    _hub("raw-train", "sentiment/imdb", P, "stanfordnlp/imdb", "e6281661ce1c48d982bc483cf8a173c1bbeb5d31", "unknown (academic)", "binary sentiment (noul)", patterns=("plain_text/train-*", "plain_text/test-*")),
    _hub("raw-train", "safety/aegis", P, "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c", "CC-BY-4.0", "devtools-v1 source: content safety", files={
        "train.json": "154fba82c71d9fa73abd2ca5588a198e693ddc816c83444df180a22f613e02f6",
        "validation.json": "a97200e226ad4f6ba6a639982f817f675909bc74513859df3e8fc9a92951dfcd",
        "test.json": "b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35",
    }),
    _hub("raw-train", "safety/aart-filter", M, "walledai/AART", "dd98454c52d34e860f5927625a3989edd5617d78", "CC-BY-4.0", "filter list: Kev drops Aegis prompts that match AART (PaLM-generated)", files={
        "data/train-00000-of-00001.parquet": "d22e3a35637a129bbffa6055a9ae83df1327f373ea22ef6f3cfa1c2d17eba0f1",
    }),
)

RAW_NEW: tuple[Item, ...] = (
    _hub("raw-new", "knowledge/arc", P, "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "CC-BY-SA-4.0", "knowledge MCQ; on Kev's TRAINABLE list; targets MMLU-Pro (0.565 vs Jev 0.840)", patterns=("ARC-Challenge/*", "ARC-Easy/*")),
    _hub("raw-new", "knowledge/openbookqa", P, "allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", "unknown on Hub (AllenAI release)", "knowledge MCQ; on Kev's TRAINABLE list", patterns=("main/*",)),
    _hub("raw-new", "knowledge/commonsenseqa", P, "tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", "MIT", "commonsense MCQ; on Kev's TRAINABLE list", patterns=("data/*",)),
    _hub("raw-new", "knowledge/qasc", P, "allenai/qasc", "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "CC-BY-4.0", "two-fact composition MCQ; targets multi-hop", patterns=("data/*",)),
    _hub("raw-new", "intent/massive-en", P, "AmazonScience/massive", "ed58ac423a2f4121720918bf5301577edce4ffd3", "CC-BY-4.0", "60 intents / 18 scenarios (en-US); routing breadth without touching clinc150; refs/convert/parquet commit", patterns=("en-US/*",)),
    _hub("raw-new", "tools/glaive-function-calling", P, "glaiveai/glaive-function-calling-v2", "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac", "Apache-2.0", "tool choice and call-or-not decisions; targets breadth-v1 tools (BFCL/ToolRet/API-Bank stay eval-only)", files={
        "glaive-function-calling-v2.json": "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad",
    }),
)

RAW_EVAL: tuple[Item, ...] = (
    _hub("raw-eval", "transfer/qnli", E, "nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "CC-BY-SA-4.0", "transfer: does the sentence answer the question", patterns=("qnli/*",)),
    _hub("raw-eval", "transfer/tweeteval-offensive", E, "cardiffnlp/tweet_eval", "b3a375baf0f409c77e6bc7aa35102b7b3534f8be", "see source", "transfer: noisy-label honesty check", patterns=("offensive/*",)),
    _hub("raw-eval", "transfer/paws", E, "google-research-datasets/paws", "161ece9501cf0a11f3e48bd356eaa82de46d6a09", "free use (Google)", "transfer: paraphrase", patterns=("labeled_final/*",)),
    _hub("raw-eval", "transfer/sciq", E, "allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815", "CC-BY-NC-3.0", "transfer: science MCQ", patterns=("data/*",)),
    _hub("raw-eval", "transfer/mmlu", E, "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "MIT", "transfer: 4-way knowledge (Kev-4B 0.725 vs Jev 0.90)", patterns=("all/test-*", "all/validation-*", "all/dev-*")),
    _hub("raw-eval", "transfer/emotion", E, "dair-ai/emotion", "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "research only", "transfer: noisy-label honesty check", patterns=("split/*",)),
    _hub("raw-eval", "transfer/mmlu-pro", E, "TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "MIT", "transfer-v9: 10-way knowledge (Kev-4B 0.565 vs Jev 0.840)", patterns=("data/*",)),
    _hub("raw-eval", "devtools/when2call", E, "nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", "CC-BY-4.0", "devtools-v1 eval-only source: when to call a tool", files={
        "test/when2call_test_mcq.jsonl": "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c",
    }),
    _hub("raw-eval", "devtools/prompt-injection-deepset", E, "deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd", "Apache-2.0", "devtools-v1 eval-only source: prompt injection", files={
        "data/train-00000-of-00001-9564e8b05b4757ab.parquet": "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
        "data/test-00000-of-00001-701d16158af87368.parquet": "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
    }),
    _hub("raw-eval", "devtools/prompt-injection-gandalf", E, "Lakera/gandalf_ignore_instructions", "04737b65e90a6794ec227012e4a255a7def6344b", "MIT", "devtools-v1 eval-only source: prompt injection", files={
        "data/train-00000-of-00001-ded53be747ff55cd.parquet": "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439",
        "data/validation-00000-of-00001-94481a2a09ff2fff.parquet": "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6",
        "data/test-00000-of-00001-bc92128b9288a6d1.parquet": "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972",
    }),
    _hub("raw-eval", "breadth/knowledge/musr", E, "TAUR-Lab/MuSR", "7c365b439a222150f317764d4f16ae6c96d7d94a", "CC-BY-4.0", "breadth-v1 source (knowledge)", files={
        "murder_mystery.csv": "ce19b6dc0b953f698e79528c72705804bc97e772a4303734808971f63ce233a7",
        "object_placements.csv": "98cd17d2c9ea53664e274365e901c90dfcaa40d17547dfbd369f1cd26fd2a81c",
        "team_allocation.csv": "ddbef3bb61857a7cc3b73db62fac4605679e1b7ff4357d57fdfcf91f7caaa1f8",
    }),
    _hub("raw-eval", "breadth/knowledge/sata-bench", E, "sata-bench/sata-bench", "ba43a7ab537adfa3498e3a160a6d1eafbefc95c1", "CC-BY-NC-4.0", "breadth-v1 source (knowledge)", files={
        "data_main.json": "d9809889057a6f37bf0dd35b371a4cf3f3a23ff431f9fc69201220598e950112",
    }),
    _url("raw-eval", "breadth/knowledge/chessbench", E, "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag", "action_value_data.bag", "5f73aac8f60e31734cdbf276ba3fca8d5ba5cb6171ba600e31af4f36327986b0", "Apache-2.0", "breadth-v1 source (knowledge)"),
    _hub("raw-eval", "breadth/language/hellaswag", E, "Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "MIT", "breadth-v1 source (language)", files={
        "data/validation-00000-of-00001.parquet": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
    }),
    _url("raw-eval", "breadth/language/contractnli", E, "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip", "contract-nli.zip", "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b", "CC-BY-4.0", "breadth-v1 source (language)"),
    _hub("raw-eval", "breadth/retrieval/clinc150", E, "clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "CC-BY-3.0", "breadth-v1 source (retrieval)", files={
        "plus/test-00000-of-00001.parquet": "3e60e45b25bf86543aa5df8ba4fcc674114164e6184f0197690648c2908d0102",
        "plus/validation-00000-of-00001.parquet": "fbd545b46c611c4a7ba4b48cae6c7f09bb5b59f33ff56206ad1cd366c85cdfaa",
        "README.md": "e28704a3c04f6b05c286ca2a8891bbf32c69c000dedfc5825d6e1713369b7a89",
    }),
    _hub("raw-eval", "breadth/retrieval/bright", E, "xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae", "CC-BY-4.0", "breadth-v1 source (retrieval)", files={
        "examples/biology-00000-of-00001.parquet": "6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829",
        "documents/biology-00000-of-00001.parquet": "8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670",
        "examples/earth_science-00000-of-00001.parquet": "5d29f108701111984eb91c93d3e340a784a99df47dc11f43783d1b994010a91d",
        "documents/earth_science-00000-of-00001.parquet": "abcb2cd130d7d333b364bf5c89b7ff3829e0f88eb6a2de8232c3df1173eeb8a2",
        "examples/economics-00000-of-00001.parquet": "2a79f0f3a881c7c03a258cf8ef8ac2db1ca9080963252d9a020bb45a264aa037",
        "documents/economics-00000-of-00001.parquet": "f3ba8a0fbc9a9aed07b4970cc686e32cfefcd06d6922402587adf871f006394c",
        "examples/psychology-00000-of-00001.parquet": "404e7dff2a4528419df0bdc162541e92138e35b78918d82d3a04ade5b8f7876b",
        "documents/psychology-00000-of-00001.parquet": "085d381739cb24b4227dfaf577f39d0adcad8b7b1ae74be028ac239d37be3c1d",
        "examples/robotics-00000-of-00001.parquet": "621484c87c9ebae12f81e32a0a8c5d085af4b95cbe1b575ab40ae4b659adb53a",
        "documents/robotics-00000-of-00001.parquet": "2c83f286006a3b2e11a677abe88f382009c5ee79f97c1f43f6a571f3f94e6d15",
        "examples/stackoverflow-00000-of-00001.parquet": "97d417ba449ef70c9c9ae2937e9df106654a2554ce1533b090cb64b998a077e1",
        "documents/stackoverflow-00000-of-00001.parquet": "d54559692f925666c3c6b1d33a696a64ef324cf5aaeff9d6f4d11fba5cd5ac8b",
        "examples/sustainable_living-00000-of-00001.parquet": "61f97837a16b47a0d9953039cf0b6a53d0fc5deae96a34f839b7cb5e798eb117",
        "documents/sustainable_living-00000-of-00001.parquet": "474628623cf9de252bd80a7d1b667aa5070e21b87e1dd33f6723db4d24121fdf",
    }),
    _url("raw-eval", "breadth/retrieval/clinc150-domains", E, "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/domains.json", "domains.json", "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a", "CC-BY-3.0", "breadth-v1 source (retrieval)", dest="breadth/retrieval/clinc150"),
    _url("raw-eval", "breadth/retrieval/sgd", E, "https://codeload.github.com/google-research-datasets/dstc8-schema-guided-dialogue/tar.gz/e852981ae34990f4358979625854259302feaa78", "dstc8-schema-guided-dialogue.tar.gz", "ff97a9ab52b4cc9f25e1a093c96431512465e8377f9a1f57dc710a10484d2188", "CC-BY-SA-4.0", "breadth-v1 source (retrieval)"),
    _hub("raw-eval", "breadth/tools/bfcl", E, "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda", "Apache-2.0", "breadth-v1 source (tools)", files={
        "BFCL_v3_simple.json": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
        "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
        "BFCL_v3_parallel.json": "fad375d68295776efb63e5aa331dd884f8635e7bb50e05908113c74f14281c3c",
        "BFCL_v3_parallel_multiple.json": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
        "BFCL_v3_irrelevance.json": "975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690",
        "BFCL_v3_live_simple.json": "b398743e6c0dfd2a87cee97a3bc1b9b90b81edc228c3153a820a93b51186fc00",
        "BFCL_v3_live_multiple.json": "af2520ac5bb0b05aa9b8abe14a79b87198e5a3137a707d4f5633c792330f0cee",
        "BFCL_v3_live_parallel.json": "6c26e9fdc3350cf596e6d1ea9c179cbff834761bccf562f4141ed29a839ca421",
        "BFCL_v3_live_parallel_multiple.json": "21d4b9319c1faac431e22757b367ea28917fe467364c3a4b17f16ec06d4f6e79",
        "BFCL_v3_live_irrelevance.json": "0d259e1c3ab6ba06c2a51911e2f49ecaa2eac3723a4a14f183a7687955e61338",
        "possible_answer/BFCL_v3_simple.json": "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
        "possible_answer/BFCL_v3_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
        "possible_answer/BFCL_v3_parallel.json": "3af54bf04e5b7640b03a9b6a82440f52474feb24b398f94435fd2b155c7727de",
        "possible_answer/BFCL_v3_parallel_multiple.json": "df4d2c16238c1465dc04697c273cf3ab4a450075fcd274e3dc86ad5bf0049b75",
        "possible_answer/BFCL_v3_live_simple.json": "4a1988780cdd8a723967587614cb84a9f53aa5d6fd37f83c78a0dd7acc2f3d15",
        "possible_answer/BFCL_v3_live_multiple.json": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
        "possible_answer/BFCL_v3_live_parallel.json": "5e3d26c9103dd25912003d3c5f681f30efc6fd2b5f8ba474450f2df21fdf6fcb",
        "possible_answer/BFCL_v3_live_parallel_multiple.json": "f5b5f360556c5feb51db46fb9f56ee4b304f4b45b161599bbb14161c98a2873f",
    }),
    _hub("raw-eval", "breadth/tools/toolret-queries", E, "mangopy/ToolRet-Queries", "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445", "see source", "breadth-v1 source (tools)", files={
        "appbench/queries-00000-of-00001.parquet": "e7efff5e12a7676df46ded343abfdd44cc3c4fbeacee3d6b89edec8a9022b5d7",
        "autotools-food/queries-00000-of-00001.parquet": "a780caf11c358db91552c6c75a7c8fe6312c73dcead3413543b9ccb1ffef6419",
        "autotools-music/queries-00000-of-00001.parquet": "aff4630b7e8828b262df047b2db1172cf24b57c15e454261359f6e64e028487d",
        "autotools-weather/queries-00000-of-00001.parquet": "80203c8ee6070ca102b08eb90b134f9602e4d085ba2f4da4f89288e1edd5f68f",
        "craft-math-algebra/queries-00000-of-00001.parquet": "70993dcb4bc2b55f42994bed7f08b785d6af6baad941ee6973db26e2700b5df6",
        "craft-tabmwp/queries-00000-of-00001.parquet": "c86458e41e26ac631aea4223fb9e24803d1b8117e31c28265824c10d221cfdf5",
        "craft-vqa/queries-00000-of-00001.parquet": "014a4d68f6f55f74820139ea6008cbce91c609d8e0965aac94ac1599683d8d9b",
        "gorilla-huggingface/queries-00000-of-00001.parquet": "ed64800b722bc8e2b8520feb3e5defc3988b0cc858929396ffc930625e71f703",
        "gorilla-pytorch/queries-00000-of-00001.parquet": "b6423e142bab08ac0d7f9cdc1851a8e85a915970936139c795f697188b137ae0",
        "gorilla-tensor/queries-00000-of-00001.parquet": "5b037823ac75e503df7abb08c51dcf7843f86ffb8ff45a625a09f1589b8f1fc7",
        "gpt4tools/queries-00000-of-00001.parquet": "7171ba30b7930b76e3562d38c0cf74c2c778fbc429e2087234f8bef577ed7576",
        "gta/queries-00000-of-00001.parquet": "757560b5e587993e8fb05c9d3ad146016cdefa21b938bd50b78ad6045067bb08",
        "metatool/queries-00000-of-00001.parquet": "f4839301cd345a6b5eef1ebfd1710dfb209e5c9b6812a31ed8d7152979393ca0",
        "mnms/queries-00000-of-00001.parquet": "d591f2cf3974abe21c6c0462a935b53b807246b95ed4d7439d5d05e0e97973ca",
        "restgpt-spotify/queries-00000-of-00001.parquet": "a2dcd32971d6e807c437f09f3c09f84a2bc53a805468b98307ea41116c35e0a4",
        "restgpt-tmdb/queries-00000-of-00001.parquet": "aa8a256d7100b005f6258f60229b5c0350967be3b8b8d6e6b58dfa79a8059193",
        "reversechain/queries-00000-of-00001.parquet": "d4da51689cfc10f045e37bce6ffeb665e43a67faa9d71834abc84106603ad854",
        "rotbench/queries-00000-of-00001.parquet": "2f75a1caf9c105560b5da02bd17e733e7d4d7e57ed6e0696728ffc36dae00159",
        "t-eval-dialog/queries-00000-of-00001.parquet": "da566b97d930ce44b89bca85c2e7fa6acac6809110f76bb2e354a77a978ed97e",
        "t-eval-step/queries-00000-of-00001.parquet": "eddd81571593a320e09b96d3807350d81f5c2778087489844d73551fca14c899",
        "taskbench-daily/queries-00000-of-00001.parquet": "7357298960afafd3afcff91559cc14e95486e729798ef1b39af044e494cb8d15",
        "taskbench-huggingface/queries-00000-of-00001.parquet": "c94f6ca056dd0906fb0f0d2cd9abe4f98a077c368fc52af9dee14325ee60d58a",
        "taskbench-multimedia/queries-00000-of-00001.parquet": "0cebfcd50a061c48c043785f80db4ec48fbaa82f764db29f20b83f82b7e6d7e4",
        "tool-be-honest/queries-00000-of-00001.parquet": "e936b9dde7b3e7256662464a118b952c3873ea0b6086d0a328a568b7b35678cb",
        "toolace/queries-00000-of-00001.parquet": "49f7c861cec1799430680cea1df6cb98a10b33227f6cb183ac32375f396fcd5a",
        "toolalpaca/queries-00000-of-00001.parquet": "627659a13d1b30a76e94d91ea27b14b4bc3be30e3fd6cd727f58ca1665bb1fe2",
        "toolbench-sam/queries-00000-of-00001.parquet": "36075db0985db103afac9ad959f7974e79eb267c7d47fd31a6de4813e3a3802f",
        "toolbench/queries-00000-of-00001.parquet": "2bb32ea009d6ece37b1b785330247aacac037a20dd4522632cb3359809c8837d",
        "toolemu/queries-00000-of-00001.parquet": "f11491359b61bfb4a6f4d9bed2c45baab547938986e749e9eee555c667887020",
        "tooleyes/queries-00000-of-00001.parquet": "c28280ed0a3cb119cc22f3f6678610bd2b7d4f9acbe7b800f0452b9a8d8a5d09",
        "toolink/queries-00000-of-00001.parquet": "6ff2d524e1e79b8af96518e9d880ddf00cb8371e1bf58475011551b74d9d4a91",
        "toollens/queries-00000-of-00001.parquet": "344e08a552508c77cb58e87dc93f7090f5c4eedba771095e5cdf6c1b12243a79",
        "ultratool/queries-00000-of-00001.parquet": "5fc5fd59cab9e9a6852817ccb4645f81d47bf4394da232404d798c1be33d36b8",
    }),
    _hub("raw-eval", "breadth/tools/toolret-tools", E, "mangopy/ToolRet-Tools", "e06c38c75612b6536bd959e08cdd345894aba6a7", "see source", "breadth-v1 source (tools)", files={
        "web/tools-00000-of-00001.parquet": "57faceebd778b2cb74571dca0e71e8bc8078c11ba05dce0efc562e818d3950e6",
        "code/tools-00000-of-00001.parquet": "a5b1a8111a40128b1429fe53a490e627fd2832ebe2e641db5eeb695ecb4f0d9a",
        "customized/tools-00000-of-00001.parquet": "e3d0cf64f5d5321751401677c820c5e8c7df79bfd67a85c01e275fe573cbde41",
    }),
    _hub("raw-eval", "breadth/tools/apibank", E, "liminghao1630/API-Bank", "12e8158b7628c168f07e8f31fbbe3445e99f44cf", "MIT", "breadth-v1 source (tools)", files={
        "test-data/level-1-api.json": "0d3155c5495399836b5b1f1cc4eb6d5d50efa16b2734af50a016a158e9dd79bb",
    }),
    _hub("raw-eval", "breadth/tools/routerbench", E, "withmartian/routerbench", "784021482c3f320c6619ed4b3bb3b41a21424fcb", "see source", "breadth-v1 source (tools)", files={
        "routerbench_0shot.pkl": "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
    }),
    _url("raw-eval", "breadth/arts/humicroedit", E, "https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip", "semeval-2020-task-7-dataset.zip", "12a6cbf28c8b698ad80be42a65ac867b57e4c71662eedab607805e167ba791ab", "none stated", "breadth-v1 source (arts)"),
    _url("raw-eval", "breadth/arts/cfcolor", E, "https://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip", "cfcolor.zip", "47c07095642cfab3c2eeab366a5d152b07783cbb5af390cbfda7d7c13db4b54c", "permissive notice", "breadth-v1 source (arts)"),
)

RAW_BULK: tuple[Item, ...] = (
    _hub("raw-bulk", "documents/cfpb-complaints", P, "davidheineman/consumer-finance-complaints-large", "44cfa170a402e254407470275ce05d7dcaccde30", "US government work (public domain)", "documents-v1 source (1.4 GB)", patterns=("data/*",)),
    _hub("raw-bulk", "devtools/commitpackft", P, "bigcode/commitpackft", "fc56fe33c030c6daa414c2b112c932b8eed085e6", "MIT (dataset); per-row repository licence", "devtools-v1 source: commit type; Kev's 12 languages", files={
        "data/python/data.jsonl": "d167da37e1058371c48e057cd8815d03700c867dd8bcf58e61420d4dcd288d73",
        "data/javascript/data.jsonl": "968688a1163c2dbbced0d0f9cf1dd52bc0b626c2ad9b02ff23f5700b2e277f25",
        "data/typescript/data.jsonl": "43d32596844c62aa23b0e5a8811648f6f20fa07d9ab80499d66507136b844a76",
        "data/java/data.jsonl": "c51a331320f4220d6053e3389485916c738fcaccfd583298b7738af834fa19b9",
        "data/go/data.jsonl": "05a65e973fcc6894a234fb382987cc25e26f053d60b032ea83540a5b36a528b8",
        "data/rust/data.jsonl": "5611f3701026acf3c43ba64da50f4f65a078e84fba56e2bf313e8f2996b0cc68",
        "data/c/data.jsonl": "a67d9aeac62e693fb782030ad7c4aa9e0d9f38a02bd2e06a438604a92628ca71",
        "data/c++/data.jsonl": "ee59eb0dca07dadb74bde829134e0d3c42176e198773f125a14554a5547df4f2",
        "data/c#/data.jsonl": "a58a0a03a9965fae11e20c53cf5661ecca6ae5b3ca241f2989db4c2c1be5c04a",
        "data/ruby/data.jsonl": "6b1970e1c286f98445e519b92360eac5618f29f422c56ac949623d42dccd04b1",
        "data/php/data.jsonl": "fda88bbb6483a504dcbaec36abd363eb6ae99531431fecc924d95ecdad91c691",
        "data/shell/data.jsonl": "08eef5271d29368db2af8b0ca127c8d249d056ae7f6bce9c0692ade5dcad66ca",
    }),
    _url("raw-bulk", "devtools/codereviewer", P, "https://zenodo.org/records/6900648/files/Diff_Quality_Estimation.zip?download=1", "Diff_Quality_Estimation.zip", "86d054de47741cb358c8a17ab55b6191356fc63e44010c864dd790481f41fff5", "CC-BY-4.0", "devtools-v1 source: did a reviewer comment on the hunk (2.8 GB)"),
    _url("raw-bulk", "devtools/flakeflagger-results", P, "https://zenodo.org/records/4450723/files/test_results.csv?download=1", "test_results.csv", "86210ed8ac0171a3d64cf5ab83d503cc97e82845e0299b9f55599a8414218f13", "CC-BY-4.0", "devtools-v1 source: flaky tests", dest="devtools/flakeflagger"),
    _url("raw-bulk", "devtools/flakeflagger-projects", P, "https://zenodo.org/records/4450723/files/Project_Info.csv?download=1", "Project_Info.csv", "f0064b25ed64995842465b67c6fec86b5ddf72d30e1a7dc5b64abd8b0208e9fd", "CC-BY-4.0", "devtools-v1 source: flaky tests", dest="devtools/flakeflagger"),
    _url("raw-bulk", "devtools/flakeflagger-code", P, "https://codeload.github.com/AlshammariA/FlakeFlagger/tar.gz/2fcbafc4713abbcc452bee07aa9681c7c8ccb707", "FlakeFlagger.tar.gz", "67da177462a3026655a9ed8ef4e7c87a86d538c599ed484080db3b355f6d76b1", "BSD-3-Clause", "devtools-v1 source: test bodies", dest="devtools/flakeflagger"),
)

ITEMS: tuple[Item, ...] = SUITES + RAW_TRAIN + RAW_NEW + RAW_EVAL + RAW_BULK
validate(ITEMS)
