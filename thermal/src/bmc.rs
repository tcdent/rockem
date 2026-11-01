//! The BMC's six chassis fan headers, through the vendor's Redfish endpoint the web UI uses
//! (found 2026-11-01, bringup/box/thermal/bmc/): `GET/PATCH /redfish/v1/Chassis/1/Oem/Public/Thermal`
//! with `{"ControlMode":"Manual","Fans":[{"PwmSensorName":"PWM1","SpeedRatio":0-100},..]}` and an
//! `If-Match` of the resource's ETag. The PWMs read back at `/redfish/v1/Chassis/1/Sensors/fanpwm_PWMn`,
//! the BMC's temperatures (VRs, inlet, outlet) at `/redfish/v1/Chassis/1/Thermal`. IPMI's Set Sensor
//! Reading is refused by this BMC; the phosphor-pid-control tables are inert while ControlMode is Manual.
//!
//! bmcweb is HTTPS only (TLS 1.3) with a self-signed certificate, so the client is rustls with no
//! roots and the certificate pinned by its SHA-256. HTTP/1.1, one connection a request, basic auth.

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig, ClientConnection, DigitallySignedStruct, SignatureScheme, StreamOwned};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Pinned(Vec<u8>);

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self, end_entity: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let got = ring::digest::digest(&ring::digest::SHA256, end_entity.as_ref());
        if got.as_ref() == self.0.as_slice() {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General("the BMC's certificate is not the pinned one".into()))
        }
    }
    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider().signature_verification_algorithms.supported_schemes()
    }
}

fn b64(s: &str) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let b = s.as_bytes();
    let mut out = String::new();
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if c.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn hex(s: &str) -> Option<Vec<u8>> {
    let s: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if s.len() != 64 {
        return None;
    }
    (0..64).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

pub struct Bmc {
    host: String,
    auth: String,
    config: Arc<ClientConfig>,
    pub last_ratio: Option<u8>,
    last_patch: Option<Instant>,
    pub taken: bool,
}

pub struct Readings {
    pub vcore_vr: Option<f64>,
    pub vsoc_vr: Option<f64>,
    pub inlet: Option<f64>,
    pub outlet: Option<f64>,
    pub cpu: Option<f64>,
}

impl Bmc {
    /// `host` is the BMC's address, `pin` its certificate's SHA-256 (any separators), `env` a file with
    /// IPMI_USER= and IPMI_PASSWORD= lines (the /etc/ipmi convention).
    pub fn new(host: &str, pin: &str, env_path: &str) -> Result<Bmc, String> {
        let pin = hex(pin).ok_or("the certificate pin is not 64 hex digits")?;
        let env = std::fs::read_to_string(env_path).map_err(|e| format!("{env_path}: {e}"))?;
        let get = |k: &str| env.lines().find_map(|l| l.strip_prefix(&format!("{k}=")).map(|v| v.trim().to_string()));
        let (user, pass) = (get("IPMI_USER").ok_or("IPMI_USER missing")?, get("IPMI_PASSWORD").ok_or("IPMI_PASSWORD missing")?);
        let config = ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| e.to_string())?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(Pinned(pin)))
            .with_no_client_auth();
        Ok(Bmc { host: host.to_string(), auth: b64(&format!("{user}:{pass}")), config: Arc::new(config), last_ratio: None, last_patch: None, taken: false })
    }

    /// One HTTP/1.1 request over a fresh TLS connection: (status, ETag, body).
    fn request(&self, method: &str, path: &str, body: Option<&str>, etag: Option<&str>) -> Result<(u16, Option<String>, String), String> {
        let name = ServerName::try_from(self.host.clone()).map_err(|e| e.to_string())?;
        let conn = ClientConnection::new(self.config.clone(), name).map_err(|e| e.to_string())?;
        let tcp = TcpStream::connect((self.host.as_str(), 443)).map_err(|e| e.to_string())?;
        tcp.set_read_timeout(Some(Duration::from_secs(10))).ok();
        tcp.set_write_timeout(Some(Duration::from_secs(10))).ok();
        let mut s = StreamOwned::new(conn, tcp);
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\nAuthorization: Basic {}\r\nAccept: application/json\r\nConnection: close\r\n", self.host, self.auth);
        if let Some(e) = etag {
            req.push_str(&format!("If-Match: {e}\r\n"));
        }
        if let Some(b) = body {
            req.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len()));
        }
        req.push_str("\r\n");
        if let Some(b) = body {
            req.push_str(b);
        }
        s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        let _ = s.read_to_end(&mut raw); // close_notify may be missing: whatever arrived is the response
        let text = String::from_utf8_lossy(&raw).into_owned();
        let (head, body) = text.split_once("\r\n\r\n").ok_or("no HTTP header in the reply")?;
        let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("no status")?;
        let etag = head.lines().find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("etag")).map(|(_, v)| v.trim().to_string()));
        Ok((status, etag, body.to_string()))
    }

    /// The number after `"<key>":` in a JSON text (bmcweb pretty-prints: spaces after the colon), the
    /// first occurrence; `"NA"` and null read as None.
    fn number(body: &str, key: &str) -> Option<f64> {
        let i = body.find(&format!("\"{key}\""))? + key.len() + 2;
        let rest = body[i..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
        let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).unwrap_or(rest.len());
        rest[..end].parse().ok()
    }

    /// The index just past `"Name": "<name>"`, spaces or not.
    fn named(body: &str, name: &str) -> Option<usize> {
        let mut from = 0;
        while let Some(i) = body[from..].find("\"Name\"") {
            let at = from + i + 6;
            let rest = body[at..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
            if let Some(v) = rest.strip_prefix('"') {
                if v.starts_with(name) && v[name.len()..].starts_with('"') {
                    return Some(at);
                }
            }
            from = at;
        }
        None
    }

    /// The BMC's own temperatures: `"Name":"CPU Vcore VR T",...,"ReadingCelsius":53.0` pairs.
    pub fn readings(&self) -> Result<Readings, String> {
        let (st, _, body) = self.request("GET", "/redfish/v1/Chassis/1/Thermal", None, None)?;
        if st != 200 {
            return Err(format!("Thermal: http {st}"));
        }
        let find = |name: &str| -> Option<f64> {
            let i = Self::named(&body, name)?;
            Self::number(&body[i..], "ReadingCelsius")
        };
        Ok(Readings { vcore_vr: find("CPU Vcore VR T"), vsoc_vr: find("CPU Vsoc VR T"), inlet: find("Inlet Temp"), outlet: find("Outlet Temp"), cpu: find("CPU Temp") })
    }

    pub fn pwm(&self, n: u8) -> Option<f64> {
        let (st, _, body) = self.request("GET", &format!("/redfish/v1/Chassis/1/Sensors/fanpwm_PWM{n}"), None, None).ok()?;
        if st != 200 {
            return None;
        }
        Self::number(&body, "Reading")
    }

    /// All six headers to one ratio, Manual mode, with the resource's current ETag.
    fn patch(&mut self, ratio: u8) -> Result<(), String> {
        let path = "/redfish/v1/Chassis/1/Oem/Public/Thermal";
        let (st, etag, _) = self.request("GET", path, None, None)?;
        if st != 200 {
            return Err(format!("GET {path}: http {st}"));
        }
        let fans: Vec<String> = (1..=6).map(|n| format!("{{\"PwmSensorName\":\"PWM{n}\",\"SpeedRatio\":{ratio}}}")).collect();
        let body = format!("{{\"ControlMode\":\"Manual\",\"Fans\":[{}]}}", fans.join(","));
        let (st, _, reply) = self.request("PATCH", path, Some(&body), etag.as_deref())?;
        if st != 200 {
            return Err(format!("PATCH {path}: http {st}: {}", reply.chars().take(160).collect::<String>()));
        }
        self.last_ratio = Some(ratio);
        self.last_patch = Some(Instant::now());
        Ok(())
    }

    pub fn take(&mut self, ratio: u8) -> Result<(), String> {
        self.patch(ratio)?;
        self.taken = true;
        Ok(())
    }

    /// Rate-limited: a move of two points or more, five seconds apart at most.
    pub fn set(&mut self, ratio: u8) -> Option<Result<(), String>> {
        if !self.taken {
            return None;
        }
        let moved = self.last_ratio.map_or(true, |l| (l as i16 - ratio as i16).abs() >= 2);
        let due = self.last_patch.map_or(true, |t| t.elapsed().as_secs_f64() >= 5.0);
        if moved && due { Some(self.patch(ratio)) } else { None }
    }

    /// The hand-back is Manual 100 %: the state tcdent ran for days, loud and safe. The BMC's automatic
    /// profile is not trusted (tcdent, 2026-10-26).
    pub fn release(&mut self) {
        if self.taken {
            let _ = self.patch(100);
            self.taken = false;
        }
    }
}
