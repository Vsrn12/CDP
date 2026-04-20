# Construccion de un cluster virtual con MPI

## Objetivo

Construir un cluster con maquinas virtuales que incluya:

- Un nodo de administracion.
- Un nodo de computacion.
- Sistema de archivos compartido mediante NFS.
- Conexion segura mediante SSH.
- Validacion del funcionamiento del cluster ejecutando `hello_mpi.c`.

## Topologia propuesta

- Nodo de administracion: `admin`
- Nodo de computacion: `compute1`
- Red privada interna: `192.168.56.0/24`
- IP sugerida para `admin`: `192.168.56.10`
- IP sugerida para `compute1`: `192.168.56.11`
- Directorio compartido por NFS: `/shared`

## Software sugerido

- VirtualBox o VMware Workstation
- Ubuntu Server 22.04 LTS en ambas maquinas virtuales
- OpenSSH Server
- NFS Kernel Server en `admin`
- NFS Common en `compute1`
- OpenMPI (`openmpi-bin`, `libopenmpi-dev`)

## Pasos de instalacion

### 1. Crear las maquinas virtuales

Crear dos maquinas virtuales Linux con al menos estas caracteristicas:

- 2 GB de RAM por VM
- 1 o 2 CPU virtuales
- Disco de 20 GB
- Adaptador de red interno o host-only para la comunicacion entre nodos

### 2. Configurar nombres e IPs

En `admin` y `compute1`, editar `/etc/hosts` para registrar ambos nodos:

```text
192.168.56.10 admin
192.168.56.11 compute1
```

Verificar conectividad con:

```bash
ping -c 4 admin
ping -c 4 compute1
```

### 3. Instalar dependencias

En ambos nodos:

```bash
sudo apt update
sudo apt install -y build-essential openssh-server openmpi-bin libopenmpi-dev nfs-common
```

Solo en `admin`:

```bash
sudo apt install -y nfs-kernel-server
```

### 4. Configurar SSH sin contrasena

En `admin`:

```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id usuario@compute1
ssh usuario@compute1 hostname
```

El ultimo comando debe conectarse sin solicitar contrasena.

### 5. Configurar NFS en el nodo de administracion

En `admin`:

```bash
sudo mkdir -p /shared
sudo chown $USER:$USER /shared
echo "/shared 192.168.56.0/24(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

### 6. Montar NFS en el nodo de computacion

En `compute1`:

```bash
sudo mkdir -p /shared
sudo mount admin:/shared /shared
df -h | grep /shared
```

Para montaje persistente, agregar esta linea a `/etc/fstab`:

```text
admin:/shared /shared nfs defaults 0 0
```

### 7. Preparar el archivo de hosts de MPI

En `admin`, crear `/shared/hosts.txt` con este contenido:

```text
admin slots=2
compute1 slots=2
```

### 8. Compilar el programa MPI

Guardar el archivo `hello_mpi.c` dentro de `/shared` y compilar desde `admin`:

```bash
cd /shared
mpicc hello_mpi.c -o hello_mpi
```

### 9. Ejecutar la validacion del cluster

Desde `admin`:

```bash
mpirun --hostfile /shared/hosts.txt -np 4 /shared/hello_mpi
```

## Salida esperada

La salida correcta debe mostrar procesos ejecutandose en ambos nodos. Un ejemplo esperado es:

```text
>> Proceso  0 de  4 activado en admin
>> Proceso  1 de  4 activado en compute1
>> Proceso  2 de  4 activado en admin
>> Proceso  3 de  4 activado en compute1
```

El orden puede variar, pero deben aparecer los nombres de ambos nodos.

## Dificultades encontradas en la instalacion

Durante una instalacion de este tipo suelen aparecer estas dificultades:

1. Problemas de red entre maquinas virtuales.
   Cuando las VMs no estan en la misma red interna o host-only, `ping`, `ssh` y `mpirun` fallan porque los nodos no pueden verse entre si.

2. Error en la resolucion de nombres.
   Si `admin` y `compute1` no estan correctamente definidos en `/etc/hosts`, MPI puede fallar al lanzar procesos remotos.

3. Acceso SSH con contrasena.
   Si no se configura autenticacion por clave publica, `mpirun` puede bloquearse o pedir contrasena al intentar iniciar procesos en el nodo remoto.

4. Permisos o exportacion incorrecta en NFS.
   Si `/shared` no fue exportado con la red correcta o no se reinicio el servicio NFS, el nodo de computacion no logra montar el directorio compartido.

5. Versiones o paquetes MPI faltantes.
   Si `mpicc` o `mpirun` no estan instalados en ambos nodos, la compilacion o la ejecucion distribuida no funcionan.

6. Restricciones de firewall.
   Algunos entornos bloquean puertos necesarios para NFS, SSH o MPI, lo que provoca fallas intermitentes en la comunicacion.

## Observacion sobre esta entrega

En este workspace se deja listo el codigo fuente `hello_mpi.c` y una guia completa de instalacion y validacion. La creacion real de las maquinas virtuales, la configuracion de red, NFS, SSH y la ejecucion distribuida deben realizarse en el hipervisor del equipo porque ese paso no puede materializarse solo con archivos dentro del proyecto.