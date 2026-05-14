use rand::{rngs::StdRng, RngCore, SeedableRng};

#[inline]
pub fn seeded(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// derive an independent child stream from a parent
#[inline]
pub fn child(parent: &mut StdRng) -> StdRng {
    StdRng::seed_from_u64(parent.next_u64())
}

// collision-resistant seed mix; replaces `seed + tid * 1000` patterns
#[inline]
pub fn mix(parts: impl IntoIterator<Item = u64>) -> u64 {
    parts
        .into_iter()
        .fold(0u64, |acc, v| splitmix64(acc ^ splitmix64(v)))
}

#[inline]
pub fn split(seed: u64, salts: impl IntoIterator<Item = u64>) -> StdRng {
    let mut s = splitmix64(seed);
    for v in salts {
        s = splitmix64(s ^ splitmix64(v));
    }
    StdRng::seed_from_u64(s)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
