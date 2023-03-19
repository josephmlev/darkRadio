#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
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
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x92997ed8, "_printk" },
	{ 0xce807a25, "up_write" },
	{ 0x642487ac, "nvidia_p2p_put_pages" },
	{ 0x37a0cba, "kfree" },
	{ 0x57bc19d2, "down_write" },
	{ 0xf42ca687, "nvidia_p2p_free_page_table" },
	{ 0x73842d37, "unmap_mapping_range" },
	{ 0xf301d0c, "kmalloc_caches" },
	{ 0x35789eee, "kmem_cache_alloc_trace" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0x9a994cf7, "current_task" },
	{ 0x50157fcc, "address_space_init_once" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0x8a35b432, "sme_me_mask" },
	{ 0x87f45924, "remap_pfn_range" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0x668b19a1, "down_read" },
	{ 0x53b954a2, "up_read" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0x5b3f3e79, "nvidia_p2p_get_pages" },
	{ 0xd6b33026, "cpu_khz" },
	{ 0x364c23ad, "mutex_is_locked" },
	{ 0xd0da656b, "__stack_chk_fail" },
	{ 0xc1352057, "__register_chrdev" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0x4fa8f1f1, "param_ops_int" },
	{ 0x541a6db8, "module_layout" },
};

MODULE_INFO(depends, "nv-p2p-dummy");


MODULE_INFO(srcversion, "21CD7488565D10BAFE3BD42");
