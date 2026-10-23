# rockem API: the inference node's scheduler (FastAPI under uvicorn). tcdent's request,
# 2026-10-23, via rockem.api; placed by infra.lead.
# Image: ~/Work/rockem/api/Containerfile, tagged with the rockem git sha.
# Listens on service-net 192.168.88.3:8765; https://rockem.a10k.co through Caddy, LAN-only (the
# API has no auth).
# EXACTLY ONE INSTANCE: the process holds the node's only queue in memory, and a second scheduler
# would believe the node free and get its connections parked. count = 1, and no canary, because
# a canary runs the new allocation beside the old one. Nomad stops the old allocation before
# starting its replacement.
job "rockem-api" {
  datacenters = ["dc1"]
  namespace   = "default"
  type        = "service"

  constraint {
    attribute = "${node.unique.name}"
    value     = "elsenor"
  }

  group "api" {
    count = 1

    network {
      port "http" {
        static       = 8765
        host_network = "service"
      }
    }

    task "api" {
      driver = "podman"
      # In-flight streams end when the process stops. Clients see the disconnect, and the node
      # is freed.
      kill_timeout = "10s"

      config {
        image        = "registry.a10k.co/rockem/api:82086278df16"
        network_mode = "host"
        ports        = ["http"]
        args         = ["--host", "${NOMAD_IP_http}", "--port", "${NOMAD_PORT_http}"]
        # The bundles directory is mounted, not the bundle itself. A rebuild that replaces the
        # bundle directory would leave a direct mount pinned to the old one, and the API would
        # send a stale manifest hash that the node refuses.
        volumes = ["/home/slomp/Work/rockem/bundles:/bundles:ro"]
      }

      env {
        ROCKEM_BUNDLE           = "/bundles/glm-5.3-flash-q4rtn"
        # metatron's inference node
        ROCKEM_NODES            = "192.168.88.6:7332"
        ROCKEM_MODEL            = "glm-5.3-flash"
        ROCKEM_TRANSCRIBE_MODEL = "voxtral-small-24b"
      }

      resources {
        cpu    = 2000
        memory = 2048
      }

      service {
        name     = "rockem-api"
        provider = "nomad"
        port     = "http"

        check {
          type     = "http"
          path     = "/health"
          interval = "15s"
          timeout  = "3s"
        }
      }
    }
  }
}
