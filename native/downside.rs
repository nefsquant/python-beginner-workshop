/// The caller must supply an aligned pointer to count valid, finite f64 values.
#[no_mangle]
pub unsafe extern "C" fn downside_deviation(values: *const f64, count: usize) -> f64 {
    if values.is_null() || count == 0 {
        return f64::NAN;
    }
    // SAFETY: native.py keeps the validated NumPy buffer alive during this call.
    let returns = unsafe { std::slice::from_raw_parts(values, count) };
    let mut total = 0.0;
    for &value in returns {
        if value < 0.0 {
            total += value * value;
        }
    }
    (total / count as f64).sqrt()
}
