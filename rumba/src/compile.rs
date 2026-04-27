use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use libloading::Library;
use pyo3::prelude::*;

use crate::artifact::CompiledArtifact;
use crate::cache::cache_key;
use crate::codegen::c::Emitter;
use crate::errors::{compilation, unsupported};
use crate::frontend::ParsedInput;
use crate::types::ScalarType;

pub(crate) fn compile_parsed_function(
    parsed: ParsedInput,
    signature: Vec<ScalarType>,
) -> PyResult<CompiledArtifact> {
    let mut emitter = Emitter::new(parsed.function, signature.clone())?;
    let (source, return_type) = emitter.emit()?;
    let key = cache_key(&parsed.metadata, &signature, &source);
    let cache_dir = env::var("RUMBA_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("rumba-cache"));
    let build_dir = cache_dir.join(&key);
    fs::create_dir_all(&build_dir)
        .map_err(|err| compilation(format!("failed to create cache directory: {err}")))?;
    let source_path = build_dir.join("module.c");
    let library_path = build_dir.join(format!("module{}", shared_suffix()?));
    fs::write(&source_path, &source)
        .map_err(|err| compilation(format!("failed to write generated C source: {err}")))?;

    let cc = env::var("CC")
        .ok()
        .or_else(|| find_executable("cc"))
        .or_else(|| find_executable("clang"))
        .or_else(|| find_executable("gcc"))
        .ok_or_else(|| compilation("no C compiler found; set CC to a working compiler"))?;

    let command = vec![
        cc,
        "-shared".to_string(),
        "-fPIC".to_string(),
        "-O2".to_string(),
        source_path.display().to_string(),
        "-o".to_string(),
        library_path.display().to_string(),
    ];

    if !library_path.exists() {
        let output = Command::new(&command[0])
            .args(&command[1..])
            .output()
            .map_err(|err| compilation(format!("failed to run C compiler: {err}")))?;
        if !output.status.success() {
            return Err(compilation(format!(
                "C compilation failed:\ncommand: {}\nstdout: {}\nstderr: {}",
                command.join(" "),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )));
        }
    }

    let library = unsafe { Library::new(&library_path) }
        .map_err(|err| compilation(format!("failed to load shared library: {err}")))?;

    Ok(CompiledArtifact {
        key,
        signature,
        return_type,
        source,
        library_path,
        compile_command: command,
        library: Arc::new(library),
    })
}

fn shared_suffix() -> PyResult<&'static str> {
    if cfg!(target_os = "macos") {
        Ok(".dylib")
    } else if cfg!(target_os = "linux") {
        Ok(".so")
    } else {
        Err(unsupported(
            "native compilation currently supports Linux and macOS",
        ))
    }
}

fn find_executable(name: &str) -> Option<String> {
    let paths = env::var_os("PATH")?;
    env::split_paths(&paths)
        .map(|path| path.join(name))
        .find(|candidate| candidate.is_file())
        .map(|candidate| candidate.display().to_string())
}
