# rockem — every target runs inside the builder image (os/run.sh); the host needs only podman.
PROFILE ?= dev
RUN = os/run.sh

.PHONY: image configure kernel kernels node initramfs release qemu kvm clean gate-dump lasenora-vm

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

node:             ## the static node binary (musl)
	$(RUN) cargo build --release --target x86_64-unknown-linux-musl -p node

initramfs: node   ## initramfs.cpio.zst for $(PROFILE)
	$(RUN) os/initramfs/build.sh $(PROFILE)

release: kernel initramfs  ## kernel + initramfs for $(PROFILE)

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
