#include "FracGen.hpp"

#include <iostream>
#include <fstream>
#include <cfloat>
#include <string>

#include <vulkan/vulkan.hpp>


// The Vulkan instance and debug callback function
static vk::Instance vkInst;
static vk::DebugReportCallbackEXT debugCallback;
// The dynamic function loader
static vk::DispatchLoaderDynamic vkFuncLoad;
// The Physical device represents the actual hardware
// The vkDevice is essentially the wrapper which gives us access to it
// We also want to know about the memory we have available
static vk::Device vkDev;
static vk::PhysicalDevice vkPhys;
static vk::PhysicalDeviceMemoryProperties vkMProps;
// We need to specify a pipeline we can send command through
// In this case, it's a purely computational pipeline
static vk::Pipeline vkPpl;
static vk::PipelineLayout vkPpLayout;
static vk::ShaderModule vkShMod;
// We submit commands throug a queue
// Commands get recorderd in a buffer and allocated in a pool
static vk::CommandPool vkCmdPool;
static vk::CommandBuffer vkCmdBuffer;
static vk::Queue vkQ;
static uint32_t vkQIdx;
// Descriptors represent resources for shaders, a bit like the binding in OpenCL
// We organize these into sets
static vk::DescriptorPool      vkDescPool;
static vk::DescriptorSet       vkDescSet;
static vk::DescriptorSetLayout vkDescLayout;

constexpr const float WORKGROUP_SIZE = 32;

static VKAPI_PTR VKAPI_CALL VkBool32 debugCallbackFn(VkDebugReportFlagsEXT      flags,
                                                     VkDebugReportObjectTypeEXT objectType,
                                                     uint64_t                   object,
                                                     size_t                     location,
                                                     int32_t                    messageCode,
                                                     const char*                pLayerPrefix,
                                                     const char*                pMessage,
                                                     void*                      pUserData);


struct vkPixFormat
{
    vkPixFormat(SDL_PixelFormat fmt)
        : Amask {fmt.Amask}
        , Rloss {fmt.Rloss}
        , Gloss {fmt.Gloss}
        , Bloss {fmt.Bloss}
        , Aloss {fmt.Aloss}
        , Rshift{fmt.Rshift}
        , Gshift{fmt.Gshift}
        , Bshift{fmt.Bshift}
        , Ashift{fmt.Ashift}
    {}

    uint32_t Amask,  Rloss,  Gloss
           , Bloss,  Aloss,  Rshift
           , Gshift, Bshift, Ashift;
};

uint32_t* readFile(uint32_t& length, const char* filename) {

        FILE* fp = fopen(filename, "rb");
        if (fp == nullptr) {
            printf("Could not find or open file: %s\n", filename);
        }

        // get file size.
        fseek(fp, 0, SEEK_END);
        long filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        long filesizepadded = long(ceil(filesize / 4.0)) * 4;

        // read file contents.
        char *str = new char[filesizepadded];
        fread(str, filesize, sizeof(char), fp);
        fclose(fp);

        // data padding.
        for (int i = filesize; i < filesizepadded; i++) {
            str[i] = 0;
        }

        length = filesizepadded;
        return (uint32_t *)str;
    }

#ifdef NDEBUG
    static constexpr const bool enableValidation = false;
#else
    static constexpr const bool enableValidation = true;
#endif

void FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    /**
      * I'm leaving "all of this" here as a way of maintaining as much
      * "parity" with the other versions as possible. However, with Vulkan it is
      * particularly obvious that:
      * 1 - This would REALLY need to be broken up in digestible chunks
      * 2 - A lot of what happens in this call should be part of constructor/destructor instead
      */
    if(format == nullptr)
    {
        return;
    }

    const vk::DeviceSize texSize{sizeof(uint32_t) * static_cast<size_t>(width) * static_cast<size_t>(height)};
    const vk::DeviceSize regSize{sizeof(double) * 4};
    const vk::DeviceSize fmtSize{sizeof(vkPixFormat)};


    try
    {
        /**
         * Create the buffer object to hold the data
         * and allocate the memory on the GPU
         */

        vk::Buffer outBuf;
        vk::BufferCreateInfo bufInfo{ {}
                                    , texSize
                                    , vk::BufferUsageFlagBits::eStorageBuffer
                                    , vk::SharingMode::eExclusive  };

        vkDev.createBuffer(&bufInfo, nullptr, &outBuf);

        vk::MemoryRequirements memReqs;
        vkDev.getBufferMemoryRequirements(outBuf, &memReqs);
        vk::MemoryAllocateInfo memInfo{memReqs.size};
        vk::MemoryPropertyFlags memFlags{vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent };

        bool foundMem = false;
        uint32_t typeBits{memReqs.memoryTypeBits};
        for( uint32_t k = 0; k < vkMProps.memoryTypeCount; k++)
        {
            const vk::MemoryType memoryType = vkMProps.memoryTypes[k];

            if (  ( typeBits & 1)
               && ( (memoryType.propertyFlags & memFlags) == memFlags)
               && ( memReqs.size < vkMProps.memoryHeaps[memoryType.heapIndex].size)      )
            {
                memInfo.memoryTypeIndex = k;
                foundMem = true;
                break;
            }
            typeBits >>= 1;
        }

        if(!foundMem)
        {
            throw std::runtime_error("Failed to find adequate memory for output on the device");
        }

        vk::DeviceMemory devVec;

        vkDev.allocateMemory(&memInfo, nullptr, &devVec);
        vkDev.bindBufferMemory(outBuf, devVec, 0);

        /**
          * Same process, but now for the region
          */

        vk::Buffer regBuf;
        vk::BufferCreateInfo regInfo{ {}
                                    , regSize
                                    , vk::BufferUsageFlagBits::eStorageBuffer
                                    , vk::SharingMode::eExclusive };
        vkDev.createBuffer(&regInfo, nullptr, &regBuf);

        vkDev.getBufferMemoryRequirements(regBuf, &memReqs);

        //Acquire and allocate the memory for the buffer
        memInfo.setAllocationSize(memReqs.size);
        typeBits = memReqs.memoryTypeBits;

        foundMem = false;
        for( uint32_t k = 0; k < vkMProps.memoryTypeCount; k++)
        {
            const vk::MemoryType memoryType = vkMProps.memoryTypes[k];

            if (  ( typeBits & 1)
               && ( (memoryType.propertyFlags & memFlags) == memFlags)
               && ( memReqs.size < vkMProps.memoryHeaps[memoryType.heapIndex].size)      )
            {
                memInfo.memoryTypeIndex = k;
                foundMem = true;
                break;
            }
            typeBits >>= 1;
        }

        if(!foundMem)
        {
            throw std::runtime_error("Failed to find adequate memory for output on the device");
        }

        vk::DeviceMemory devReg;
        vkDev.allocateMemory(&memInfo, nullptr, &devReg);
        vkDev.bindBufferMemory(regBuf, devReg, 0);

        void* mem = nullptr;
        vkDev.mapMemory(devReg, 0, regSize, {} , &mem);
        memcpy(mem, &r, regSize);
        vkDev.unmapMemory(devReg);

        /**
          * And again for the PixelFormat
          */

        vk::Buffer fmtBuf;
        vk::BufferCreateInfo fmtInfo{ {}
                                    , fmtSize
                                    , vk::BufferUsageFlagBits::eStorageBuffer
                                    , vk::SharingMode::eExclusive };
        vkDev.createBuffer(&fmtInfo, nullptr, &fmtBuf);

        vkDev.getBufferMemoryRequirements(fmtBuf, &memReqs);

        //Acquire and allocate the memory for the buffer
        memInfo.setAllocationSize(memReqs.size);
        typeBits = memReqs.memoryTypeBits;

        foundMem = false;
        for( uint32_t k = 0; k < vkMProps.memoryTypeCount; k++)
        {
            const vk::MemoryType memoryType = vkMProps.memoryTypes[k];

            if (  ( typeBits & 1)
               && ( (memoryType.propertyFlags & memFlags) == memFlags)
               && ( memReqs.size < vkMProps.memoryHeaps[memoryType.heapIndex].size)      )
            {
                memInfo.memoryTypeIndex = k;
                foundMem = true;
                break;
            }
            typeBits >>= 1;
        }

        if(!foundMem)
        {
            throw std::runtime_error("Failed to find adequate memory for output on the device");
        }

        vk::DeviceMemory devFmt;
        vkDev.allocateMemory(&memInfo, nullptr, &devFmt);
        vkDev.bindBufferMemory(fmtBuf, devFmt, 0);

        mem = nullptr;
        vkPixFormat fmt{*format};
        vkDev.mapMemory(devFmt, 0, fmtSize, {} , &mem);
        memcpy(mem, &fmt, fmtSize);
        vkDev.unmapMemory(devFmt);

        /**
          * Create the descriptor set for the shader inputs
          * First - define the layout for each binding
          */

        std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding{
            //Output Vector
            {
                0,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            },
            //Region
            {
                1,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            },
            //Output Vector
            {
                2,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            }
        };

        vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo { {}
                                                            , 3
                                                            , descSetLayoutBinding.data() };
        vkDev.createDescriptorSetLayout(&descSetLayoutInfo, nullptr,&vkDescLayout);

        /**
         * Now allocate the actual descriptor set and pool
         */

        vk::DescriptorPoolSize vkPoolSize{vk::DescriptorType::eStorageBuffer, 3};
        vk::DescriptorPoolCreateInfo vkPoolInfo{ {}, 1, 1, &vkPoolSize};
        vkDev.createDescriptorPool(&vkPoolInfo, nullptr, &vkDescPool);

        vk::DescriptorSetAllocateInfo vkSetAll{vkDescPool, 1, &vkDescLayout};
        vkDev.allocateDescriptorSets( &vkSetAll, &vkDescSet);

        vk::DescriptorBufferInfo outDescBufInfo{ outBuf, 0, texSize};
        vk::DescriptorBufferInfo regDescBufInfo{ regBuf, 0, regSize};
        vk::DescriptorBufferInfo fmtDescBufInfo{ fmtBuf, 0, fmtSize};

        vk::WriteDescriptorSet outWriteDescSet{ vkDescSet
                                              , 0
                                              , 0
                                              , 1
                                              , vk::DescriptorType::eStorageBuffer
                                              , nullptr
                                              , &outDescBufInfo};

        vk::WriteDescriptorSet regWriteDescSet{ vkDescSet
                                              , 1
                                              , 0
                                              , 1
                                              , vk::DescriptorType::eStorageBuffer
                                              , nullptr
                                              , &regDescBufInfo};

        vk::WriteDescriptorSet fmtWriteDescSet{ vkDescSet
                                              , 2
                                              , 0
                                              , 1
                                              , vk::DescriptorType::eStorageBuffer
                                              , nullptr
                                              , &fmtDescBufInfo};
        vkDev.updateDescriptorSets(outWriteDescSet.descriptorCount, &outWriteDescSet, 0, nullptr);
        vkDev.updateDescriptorSets(regWriteDescSet.descriptorCount, &regWriteDescSet, 0, nullptr);
        vkDev.updateDescriptorSets(fmtWriteDescSet.descriptorCount, &fmtWriteDescSet, 0, nullptr);

        /**
         * Time to create the compute pipeline
         */

        uint32_t fileLength = 0;
        uint32_t* shaderCode = nullptr;

        shaderCode = readFile(fileLength, "FracGen.spv");

        vk::ShaderModuleCreateInfo vkShaderInfo { {}, fileLength, shaderCode };
        vkDev.createShaderModule(&vkShaderInfo, nullptr, &vkShMod);

        if(shaderCode != nullptr)
        {
            delete[] shaderCode;
        }

        /**
         * Create the pipeline itself. For compute it is one stage: just the compute shader
         */

        vk::PipelineShaderStageCreateInfo shStageCreate{ {}
                                                       , vk::ShaderStageFlagBits::eCompute
                                                       , vkShMod
                                                       , "main"};

        vk::PipelineLayoutCreateInfo ppLayoutCreate{ {}
                                                   , 1
                                                   , &vkDescLayout};

        vkDev.createPipelineLayout(&ppLayoutCreate, nullptr, &vkPpLayout);

        vk::ComputePipelineCreateInfo pplCreate{ {} , shStageCreate, vkPpLayout};
        vkPpl = vkDev.createComputePipeline({}, pplCreate);

        /**
         * Almost there, time to create the command pool and the command buffer
         */

        vk::CommandPoolCreateInfo poolInfo{ {}, vkQIdx};
        vkDev.createCommandPool(&poolInfo, nullptr, &vkCmdPool);

        vk::CommandBufferAllocateInfo cmdBufInfo{ vkCmdPool
                                                // Primary buffers can be directly submitted
                                                // Secondary buffers are chained from other buffers
                                                , vk::CommandBufferLevel::ePrimary
                                                , 1};

        vkDev.allocateCommandBuffers(&cmdBufInfo, &vkCmdBuffer);

        /**
         * Start actually using the command buffer (almost actually doing stuff)
         */

        vkCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        //Bind the pipeline AND decriptor set to the command buffer

        vkCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, vkPpl);
        vkCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, vkPpLayout, 0, 1, &vkDescSet, 0, nullptr);

        /**
         * Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
         * The number of workgroups is specified in the arguments.
         */

    //    uint32_t x = static_cast<uint32_t>( ceil(width  / WORKGROUP_SIZE));
    //    uint32_t y = static_cast<uint32_t>( ceil(height / WORKGROUP_SIZE));
    //    uint32_t workgroupCount = x*y;

        vkCmdBuffer.dispatch( static_cast<uint32_t>( ceil(width  / WORKGROUP_SIZE))
                            , static_cast<uint32_t>( ceil(height / WORKGROUP_SIZE))
                            , 1);

        vkCmdBuffer.end();

        /**
         * At last, we actually get to run something
         */

        vk::SubmitInfo subInfo{0, nullptr, nullptr, 1, &vkCmdBuffer};

        //We need a manual fence to make sure we wait for computation to end
        vk::Fence fence{};
        vk::FenceCreateInfo fenceInfo{};

       vkDev.createFence(&fenceInfo, nullptr, &fence);
       vkQ.submit(1, &subInfo, fence);
       vkDev.waitForFences(1, &fence, VK_TRUE, 100000000000);
       vkDev.destroyFence(fence, nullptr);

        /**
         * With the calculation done, time to retrieve the data
         */

        mem = nullptr;
        vkDev.mapMemory(devVec, 0, texSize, {}, &mem);
        memcpy(v,mem,texSize);
        vkDev.unmapMemory(devVec);

        vkDev.free(vkCmdPool,1,&vkCmdBuffer);
        vkDev.destroy(vkCmdPool, nullptr);
        vkDev.destroy(vkPpl, nullptr);
        vkDev.destroy(vkPpLayout, nullptr);
        vkDev.destroy(vkShMod, nullptr);
        vkDev.destroy(vkDescPool,nullptr);
        vkDev.destroy(vkDescLayout,nullptr);
        vkDev.free(devVec, nullptr);
        vkDev.free(devReg, nullptr);
        vkDev.free(devFmt, nullptr);
        vkDev.destroy(outBuf, nullptr);
        vkDev.destroy(regBuf, nullptr);
        vkDev.destroy(fmtBuf, nullptr);
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what();
    }

}

VKAPI_ATTR VkBool32 debugCallbackFn(    VkDebugReportFlagsEXT /*flags*/,
                                        VkDebugReportObjectTypeEXT /*objectType*/,
                                        uint64_t /*object*/,
                                        size_t /*location*/,
                                        int32_t /*messageCode*/,
                                        const char *pLayerPrefix,
                                        const char *pMessage,
                                        void */*pUserData*/                       )
{
    std::cout << "Vulkan Debug Report: "<< pLayerPrefix << ": " << pMessage << std::endl;
    return VK_FALSE;
}

FracGen::FracGen(bool bench)
{
    static bool once = false;
    try
    {
        vk::ApplicationInfo appInfo( "Toybrot Vulkan"
                                   , VK_MAKE_VERSION(1,0,0)
                                   , "No Engine"
                                   , VK_MAKE_VERSION(0,0,0)
                                   , VK_API_VERSION_1_1 );


        // Check for validation layer
        std::vector<const char*> enabledLayers;
        std::vector<const char*> enabledExtensions;

        if(enableValidation)
        {
            std::vector<vk::LayerProperties> layerProperties{vk::enumerateInstanceLayerProperties()};

            /*
            And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
            */
            bool foundLayer = false;
            for (vk::LayerProperties prop : layerProperties)
            {
                if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0)
                {
                    foundLayer = true;
                    break;
                }
            }

            if (!foundLayer)
            {
                throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
            }
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");

            std::vector<vk::ExtensionProperties> extensions{vk::enumerateInstanceExtensionProperties()};

            bool foundExtension = false;
            if(extensions.size() > 0)
            {
                for(const auto& ext : extensions)
                {
                    if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, ext.extensionName) == 0)
                    {
                        foundExtension = true;
                        break;
                    }
                }
            }

            if (!foundExtension)
            {
                throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
            }
            else
            {
                enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            }
        }

        //Time to create the instance

        vk::InstanceCreateInfo createInfo( vk::InstanceCreateFlags()
                                         , &appInfo
                                         , static_cast<uint>(enabledLayers.size())
                                         , enabledLayers.data()
                                         , static_cast<uint>(enabledExtensions.size())
                                         , enabledExtensions.data());


        vk::createInstance(&createInfo, nullptr, &vkInst);

        vkFuncLoad.init(vkInst);

        if(enabledLayers.size() > 0)
        {

            vk::DebugReportFlagsEXT dbgFlags( vk::DebugReportFlagBitsEXT::eError
                                            | vk::DebugReportFlagBitsEXT::eWarning
                                            | vk::DebugReportFlagBitsEXT::ePerformanceWarning );

            vk::DebugReportCallbackCreateInfoEXT cbCreateInfo(dbgFlags, &debugCallbackFn);

            // We have to try and load this function at runtime

            if (vkFuncLoad.vkCreateDebugReportCallbackEXT == nullptr)
            {
                throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
            }

            // Create and register callback.
            vkInst.createDebugReportCallbackEXT(&cbCreateInfo
                                               , nullptr
                                               , &debugCallback
                                               , vkFuncLoad);
        }

        std::vector<vk::PhysicalDevice> vkDevices{vkInst.enumeratePhysicalDevices()};

        bool foundDevice = false;
        uint32_t deviceIndex = 0;

        if(vkDevices.size() > 0)
        {
            vk::PhysicalDeviceProperties props;

            for(uint32_t i = 0; i < vkDevices.size(); i++)
            {
                vkDevices[i].getProperties(&props);
                if(!bench || !once)
                {
                    std::cout << "Device "<< i << " Name: " << props.deviceName << std::endl;
                    std::cout << "Supported version:" << VK_VERSION_MAJOR(props.apiVersion) << "."
                                                      << VK_VERSION_MINOR(props.apiVersion) << "."
                                                      << VK_VERSION_PATCH(props.apiVersion)
                                                      << std::endl;
                }
            }

            for(uint32_t i = 0; (i < vkDevices.size()) && (!foundDevice); i++)
            {
                vkDevices[i].getProperties(&props);

                std::vector< vk::QueueFamilyProperties > queueProps{vkDevices[i].getQueueFamilyProperties()};

                //Find an exclusive compute queue for the device;
                for(uint32_t j = 0; j < queueProps.size(); j++)
                {
                    if ( /*! (queueProps[j].queueFlags & vk::QueueFlagBits::eGraphics)
                        &&*/ (queueProps[j].queueFlags & vk::QueueFlagBits::eCompute )  )
                    {
                      deviceIndex = i;
                      vkQIdx = j;
                      foundDevice = true;
                      vkPhys = vkDevices[i];
                      break;
                    }
                }

                if((foundDevice) && (!bench || !once) )
                {
                    once = true;
                    std::cout << vkDevices.size() << " available Vulkan devices" << std::endl;

                    std::cout << "Using device " << i << ":" << std::endl;
                    std::cout << "Name: " << props.deviceName << std::endl;
                    std::cout << "DeviceID: " << props.deviceID << std::endl;
                    std::cout << "VendorID: " << props.vendorID << std::endl;
                    std::cout << "Supported version:" << VK_VERSION_MAJOR(props.apiVersion) << "."
                                                      << VK_VERSION_MINOR(props.apiVersion) << "."
                                                      << VK_VERSION_PATCH(props.apiVersion)
                                                      << std::endl;
                    std::cout << "Max Workgroup size:" << props.limits.maxComputeWorkGroupSize[0]  << ","
                                                       << props.limits.maxComputeWorkGroupSize[1]  << ","
                                                       << props.limits.maxComputeWorkGroupSize[2]  << std::endl;
                    std::cout << "Max Workgroup count:"<< props.limits.maxComputeWorkGroupCount[0] << ","
                                                       << props.limits.maxComputeWorkGroupCount[1] << ","
                                                       << props.limits.maxComputeWorkGroupCount[2] << std::endl;
                }

            }
        }

        if(!foundDevice)
        {
            throw std::runtime_error("Could not get a Vulkan compute queue!");
        }

        float priority = 1.f;
        vk::DeviceQueueCreateInfo queueInfo{ {}
                                            , vkQIdx
                                            , 1
                                            , &priority};

        vk::PhysicalDeviceFeatures devFeatures{};
        devFeatures.setShaderFloat64(true);

        vk::DeviceCreateInfo deviceInfo{ {} , 1 , &queueInfo };
        deviceInfo.setPEnabledFeatures(&devFeatures);

        vkPhys.createDevice(&deviceInfo, nullptr, &vkDev);
        vkDev.getQueue(vkQIdx, 0, &vkQ);
        vkPhys.getMemoryProperties(&vkMProps);
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what();
    }
}

FracGen::~FracGen()
{
    if (enableValidation)
    {
        // destroy callback.
        vkInst.destroyDebugReportCallbackEXT(debugCallback, nullptr, vkFuncLoad);
    }

    vkDev.destroy();
    vkInst.destroy();
}

bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
