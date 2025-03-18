#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x30ff7695, "module_layout" },
	{ 0xf9418c06, "param_ops_int" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0x23295fe0, "__register_chrdev" },
	{ 0xd0da656b, "__stack_chk_fail" },
	{ 0x970adefe, "nvidia_p2p_get_pages_persistent" },
	{ 0x364c23ad, "mutex_is_locked" },
	{ 0xd6b33026, "cpu_khz" },
	{ 0x5b3f3e79, "nvidia_p2p_get_pages" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0x53b954a2, "up_read" },
	{ 0x668b19a1, "down_read" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0xdb8acb90, "remap_pfn_range" },
	{ 0x8a35b432, "sme_me_mask" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x631d6bbe, "address_space_init_once" },
	{ 0xa22a96f7, "current_task" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0x4f00afd3, "kmem_cache_alloc_trace" },
	{ 0xac1c4313, "kmalloc_caches" },
	{ 0x14a2a34c, "unmap_mapping_range" },
	{ 0xf42ca687, "nvidia_p2p_free_page_table" },
	{ 0x57bc19d2, "down_write" },
	{ 0xacdf3914, "nvidia_p2p_put_pages_persistent" },
	{ 0x37a0cba, "kfree" },
	{ 0x642487ac, "nvidia_p2p_put_pages" },
	{ 0xce807a25, "up_write" },
	{ 0x92997ed8, "_printk" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "nv-p2p-dummy");


MODULE_INFO(srcversion, "B55D7686ABDA82419DDC88B");
