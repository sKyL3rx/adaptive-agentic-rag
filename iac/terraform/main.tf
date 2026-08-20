terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=4.1.0"
    }
  }
}

provider "azurerm" {
  subscription_id                 = var.subscription_id
  tenant_id                       = var.tenant_id
  resource_provider_registrations = "none"
  features {}
}


resource "azurerm_resource_group" "rg" {
  name     = "${var.project_id}-resources-2"
  location = var.region
}

resource "azurerm_kubernetes_cluster" "my_aks" {
  name                = "${var.project_id}-aks-cluster"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "${var.project_id}-k8s"


  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D4s_v3"

  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = "Lab-Student"
  }
}