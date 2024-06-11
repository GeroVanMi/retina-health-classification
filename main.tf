terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

provider "google" {
  project = "algorithmic-quartet"
  zone    = "europe-west1-b"
}

resource "google_compute_network" "vpc_network" {
  name = "terraform-network"
}

resource "google_compute_firewall" "ssh" {
  name    = "ssh-firewall"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  target_tags   = ["terraform-instance"]
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_instance" "vm_instance" {
  name         = "gpu-instance"
  machine_type = "g2-standard-4"
  tags         = ["terraform-instance"]

  boot_disk {
    initialize_params {
      size  = 80
      image = "deeplearning-platform-release/common-cu118-v20240514-ubuntu-2004"
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    provisioning_model  = "SPOT"
    preemptible         = true
    automatic_restart   = false
  }

  network_interface {
    network = google_compute_network.vpc_network.name
    access_config {
    }
  }
}
