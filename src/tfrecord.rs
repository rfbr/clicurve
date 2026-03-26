/// TFRecord file reader.
///
/// Each record in a TFRecord file has the format:
///   u64    length (little-endian)
///   u32    masked_crc32c(length bytes)
///   [u8]   data (length bytes)
///   u32    masked_crc32c(data)
///
/// Masked CRC: ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)
use anyhow::{bail, Context, Result};
use std::io::Read;

fn masked_crc(data: &[u8]) -> u32 {
    let crc = crc32c::crc32c(data);
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282_ead8)
}

pub struct TfRecordReader<R: Read> {
    reader: R,
}

impl<R: Read> TfRecordReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: Read> Iterator for TfRecordReader<R> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read 8-byte length
        let mut len_buf = [0u8; 8];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e.into())),
        }

        let result = (|| -> Result<Vec<u8>> {
            // Read and verify length CRC
            let mut len_crc_buf = [0u8; 4];
            self.reader
                .read_exact(&mut len_crc_buf)
                .context("reading length CRC")?;
            let len_crc = u32::from_le_bytes(len_crc_buf);
            let expected_len_crc = masked_crc(&len_buf);
            if len_crc != expected_len_crc {
                bail!("length CRC mismatch: got {len_crc:#x}, expected {expected_len_crc:#x}");
            }

            // Read data
            let length = u64::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; length];
            self.reader.read_exact(&mut data).context("reading data")?;

            // Read and verify data CRC
            let mut data_crc_buf = [0u8; 4];
            self.reader
                .read_exact(&mut data_crc_buf)
                .context("reading data CRC")?;
            let data_crc = u32::from_le_bytes(data_crc_buf);
            let expected_data_crc = masked_crc(&data);
            if data_crc != expected_data_crc {
                bail!("data CRC mismatch: got {data_crc:#x}, expected {expected_data_crc:#x}");
            }

            Ok(data)
        })();

        Some(result)
    }
}
