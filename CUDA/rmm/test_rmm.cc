#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    // Construct a resource that uses a coalescing best-fit pool allocator
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
    rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points

    auto p = mr->allocate(100);
    mr->deallocate(p, 100);

    return 0;
}
