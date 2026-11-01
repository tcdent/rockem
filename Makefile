# rockem — every target runs inside the builder image (os/run.sh); the host needs only podman.
PROFILE ?= dev
RUN = os/run.sh

.PHONY: image configure kernel kernels kernels-glm node thermal initramfs release publish qemu kvm clean gate-dump lasenora-vm

image:            ## build the pinned builder image
	os/builder/build-image.sh

configure:        ## .config for $(PROFILE) from the lasenora baseline + fragments
	$(RUN) os/kernel/configure.sh $(PROFILE)

kernel:           ## bzImage for $(PROFILE) -> os/release/staging/$(PROFILE)/
	$(RUN) os/kernel/build.sh $(PROFILE)

kernels:          ## the gfx1100 code objects (release + stats) on the host's nightly, then the static report (bringup/kernel_report.py): silent unless something moved, fails on scratch or a changed barrier count
	cd models/deepseek-v2-lite/gpu && cargo +nightly build --release
	cd models/deepseek-v2-lite/gpu && cargo +nightly build --release --features stats --target-dir target-stats
	cp models/deepseek-v2-lite/gpu/target-stats/amdgcn-amd-amdhsa/release/deepseek_v2_lite_gpu.elf bundles/kernels-stats.gfx1100.elf
	python3 bringup/kernel_report.py models/deepseek-v2-lite/gpu/target/amdgcn-amd-amdhsa/release/deepseek_v2_lite_gpu.elf

kernels-glm:      ## GLM-5.3-Flash's gfx1100 code object and its static report
	cd models/glm-5.3-flash/gpu && cargo +nightly build --release
	python3 bringup/kernel_report.py models/glm-5.3-flash/gpu/target/amdgcn-amd-amdhsa/release/glm_5_3_flash_gpu.elf

node:             ## the static node binary (musl); NODE_FEATURES="--features track-launches" adds a prow category, "--no-default-features" records nothing
	$(RUN) cargo build --release --target x86_64-unknown-linux-musl -p node $(NODE_FEATURES)

thermal:          ## the static thermal daemon (musl), started by init beside the node (docs/queue/101-fan-control.md)
	$(RUN) env CC_x86_64_unknown_linux_musl=gcc AR_x86_64_unknown_linux_musl=ar cargo build --release --target x86_64-unknown-linux-musl -p thermal

initramfs: node thermal   ## initramfs.cpio.zst for $(PROFILE)
	$(RUN) os/initramfs/build.sh $(PROFILE)

knobs:            ## knobs.efi: the firmware knobs (os/firmware/knobs.toml) applied ahead of the kernel; the host's toolchain (x86_64-unknown-uefi)
	cd os/firmware/knobs && cargo build --release --target x86_64-unknown-uefi
	cp os/firmware/knobs/target/x86_64-unknown-uefi/release/knobs.efi os/release/staging/$(PROFILE)/knobs.efi

release: kernel initramfs knobs  ## kernel + initramfs + knobs.efi for $(PROFILE)

publish:          ## push the staged $(PROFILE) release to registry.a10k.co/rockem/os:latest as an OCI artifact (os/release/publish.sh)
	os/release/publish.sh $(PROFILE)

qemu:             ## boot the staged $(PROFILE) release on a serial console (ctrl-a x to quit); BUNDLE=bundles/<name> shares a bundle in
	BUNDLE=$(BUNDLE) MEM=$(MEM) SMP=$(SMP) $(RUN) os/qemu.sh $(PROFILE) $(ARGS)

kvm:              ## dev host only: let this user open /dev/kvm so the container's QEMU runs the guest natively instead of under TCG.
	# Temporary, host-side, one-time: QEMU is the dev rung's stand-in for real hardware, and TCG emulates AVX2 an
	# instruction at a time. Nothing about this reaches the production image. Re-login after running it.
	sudo usermod -aG kvm $$USER

clean:
	rm -rf os/release/staging os/initramfs/stage target

gate-dump:        ## opt-in: on lasenora, batched vs sequential prefill of one prompt must dump byte-identical traces (bringup/gates/dump_diff.sh); BUNDLE=<dir on lasenora>
	ssh lasenora.local /mnt/elsenor-home/Work/rockem/bringup/gates/dump_diff.sh '$(or $(BUNDLE),$$HOME/bundles/v2lite-q4gptq)'

lasenora-vm:      ## third rung: boot $(PROFILE) in a VM on lasenora with the 7900 XT passed through (see os/lasenora-vm.sh hazard)
	ssh lasenora.local /mnt/elsenor-home/Work/rockem/os/lasenora-vm.sh $(PROFILE) '$(ARGS)'
