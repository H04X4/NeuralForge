use astra_core::{BackendKind, Error, HardwareInventory, Result};

pub fn discover() -> Result<HardwareInventory> {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let logical_cpus = num_cpus()?;
    let physical_cpus = physical_cpus()?;
    let total_ram_bytes = total_ram()?;
    let available_ram_bytes = available_ram()?;
    let storage_free_bytes = free_disk()?;

    let mut backends = vec![BackendKind::Cpu];
    detect_gpu_backends(&mut backends);

    Ok(HardwareInventory {
        os,
        arch,
        logical_cpus,
        physical_cpus,
        total_ram_bytes,
        available_ram_bytes,
        storage_free_bytes,
        backends,
    })
}

fn num_cpus() -> Result<usize> {
    Ok(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

fn physical_cpus() -> Result<usize> {
    Ok(num_cpus().unwrap_or(1))
}

#[cfg(target_os = "macos")]
fn total_ram() -> Result<u64> {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize".as_ptr();
    let rc = unsafe { libc::sysctlbyname(name, &mut size as *mut _ as *mut _, &mut len, std::ptr::null_mut(), 0) };
    if rc != 0 {
        return Err(Error::Other("failed to query hw.memsize".into()));
    }
    Ok(size)
}

#[cfg(not(target_os = "macos"))]
fn total_ram() -> Result<u64> {
    Err(Error::Other("total_ram not implemented for this OS".into()))
}

#[cfg(target_os = "macos")]
fn available_ram() -> Result<u64> {
    let total = total_ram().unwrap_or(8_000_000_000);
    let page_size = page_size();
    #[allow(deprecated)]
    let available = unsafe {
        let port = libc::mach_host_self();
        let mut count: u32 = libc::HOST_VM_INFO64_COUNT;
        let mut stats = std::mem::zeroed::<libc::vm_statistics64_data_t>();
        let ret = libc::host_statistics64(port, libc::HOST_VM_INFO64, &mut stats as *mut _ as *mut _, &mut count);
        if ret == 0 {
            let free_bytes = stats.free_count as u64 * page_size;
            let inactive_bytes = stats.inactive_count as u64 * page_size;
            let compressed_bytes = stats.compressor_page_count as u64 * page_size;
            free_bytes + inactive_bytes + compressed_bytes
        } else if total > 2_000_000_000 {
            total * 55 / 100
        } else {
            total * 60 / 100
        }
    };
    Ok(available.min(total))
}

#[cfg(target_os = "macos")]
fn page_size() -> u64 {
    let mut pagesize: u32 = 4096;
    let mut len = std::mem::size_of::<u32>();
    let name = c"hw.pagesize".as_ptr();
    let rc = unsafe {
        libc::sysctlbyname(
            name,
            &mut pagesize as *mut _ as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 { 16384 } else { pagesize as u64 }
}

#[cfg(not(target_os = "macos"))]
fn available_ram() -> Result<u64> {
    Ok(total_ram().unwrap_or(8_000_000_000) * 55 / 100)
}

fn free_disk() -> Result<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
        let path = std::ffi::CString::new(".").map_err(|e| Error::Other(e.to_string()))?;
        let rc = unsafe { libc::statfs(path.as_ptr(), &mut stat) };
        if rc == 0 {
            return Ok(stat.f_bsize as u64 * stat.f_bavail as u64);
        }
    }
    Err(Error::Other("free_disk not implemented for this OS".into()))
}

#[cfg(target_os = "macos")]
fn detect_gpu_backends(backends: &mut Vec<BackendKind>) {
    if std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists() {
        backends.push(BackendKind::Metal);
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_gpu_backends(_backends: &mut Vec<BackendKind>) {}


