use std::fmt::Write as _;

use crate::frontend::CodeMetadata;
use crate::types::ScalarType;
use crate::VERSION;

pub(crate) fn cache_key(metadata: &CodeMetadata, signature: &[ScalarType], source: &str) -> String {
    let mut payload = String::new();
    write!(&mut payload, "bytecode={:x?};", metadata.bytecode)
        .expect("write to String cannot fail");
    write!(
        &mut payload,
        "consts={};names={};python={};platform={};rumba={};",
        metadata.consts, metadata.names, metadata.python, metadata.platform, VERSION
    )
    .expect("write to String cannot fail");
    payload.push_str("signature=");
    for typ in signature {
        payload.push_str(typ.name());
        payload.push(',');
    }
    payload.push_str(";source=");
    payload.push_str(source);
    stable_hex_hash(payload.as_bytes())
}

fn stable_hex_hash(bytes: &[u8]) -> String {
    let mut first = 0xcbf29ce484222325_u64;
    let mut second = 0x84222325cbf29ce4_u64;

    for byte in bytes {
        first ^= u64::from(*byte);
        first = first.wrapping_mul(0x100000001b3);

        second ^= u64::from(byte.rotate_left(1));
        second = second.wrapping_mul(0x100000001b3);
    }

    format!("{first:016x}{second:016x}")
}
