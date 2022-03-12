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
/* The Physical device represents the actual hardware
 * The vkDevice is essentially the wrapper which gives us access to it
 * We also want to know about the memory we have available
 */
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

static std::string shaderName = "";
static bool exclusiveCompute = false;

/*
 * The validation layer itself is very useful but slows down
 * execution a lot. Additionally, further down the code I make
 * use of this one ifdef and treat if(enableValidation) pretty
 * much as "if this has been built in debug" for example, device
 * listing is more verbose and spits out available extensions and layers
 */
#ifdef NDEBUG
    static constexpr const bool enableValidation = false;
#else
    static constexpr const bool enableValidation = true;
#endif

constexpr const tbFPType WORKGROUP_SIZE = 16;

static VKAPI_PTR VKAPI_CALL VkBool32 debugCallbackFn(VkDebugReportFlagsEXT      flags,
                                                     VkDebugReportObjectTypeEXT objectType,
                                                     uint64_t                   object,
                                                     size_t                     location,
                                                     int32_t                    messageCode,
                                                     const char*                pLayerPrefix,
                                                     const char*                pMessage,
                                                     void*                      pUserData);

/******************************************************************************
 *
 * Vulkan device side data structs and helper functions
 *
 ******************************************************************************/

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


/******************************************************************************
 *
 * Vulkan setup and kernel call
 *
 ******************************************************************************/

void FracGen::Generate() const
{

    if(outBuffer->size() != cam->ScreenWidth()*cam->ScreenHeight())
    {
        outBuffer->resize(cam->ScreenWidth()*cam->ScreenHeight());
    }

    /**
      * I'm leaving "all of this" here as a way of maintaining as much
      * "parity" with the other versions as possible. However, with Vulkan it is
      * particularly obvious that:
      * 1 - This would REALLY need to be broken up in digestible chunks
      * 2 - A lot of what happens in this call should be part of constructor/destructor instead
      */


    const vk::DeviceSize texSize{outSize() };
    const vk::DeviceSize camSize{sizeof(Camera<tbFPType>) };
    const vk::DeviceSize paramSize{sizeof(Parameters<tbFPType>) };


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
          * Same process, but now for the Camera
          */

        vk::Buffer camBuf;
        vk::BufferCreateInfo camInfo{ {}
                                    , camSize
                                    , vk::BufferUsageFlagBits::eUniformBuffer
                                    , vk::SharingMode::eExclusive };
        vkDev.createBuffer(&camInfo, nullptr, &camBuf);

        vkDev.getBufferMemoryRequirements(camBuf, &memReqs);

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

        vk::DeviceMemory devCam;
        vkDev.allocateMemory(&memInfo, nullptr, &devCam);
        vkDev.bindBufferMemory(camBuf, devCam, 0);

        void* mem = nullptr;
        vkDev.mapMemory(devCam, 0, camSize, {} , &mem);
        memcpy(mem, cam.get(), camSize);

        vkDev.unmapMemory(devCam);

        /**
          * And once again, this time for the parameters
          */

        vk::Buffer paramBuf;
        vk::BufferCreateInfo paramInfo{ {}
                                    , paramSize
                                    , vk::BufferUsageFlagBits::eUniformBuffer
                                    , vk::SharingMode::eExclusive };
        vkDev.createBuffer(&paramInfo, nullptr, &paramBuf);

        vkDev.getBufferMemoryRequirements(paramBuf, &memReqs);

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

        vk::DeviceMemory devParams;
        vkDev.allocateMemory(&memInfo, nullptr, &devParams);
        vkDev.bindBufferMemory(paramBuf, devParams, 0);

        mem = nullptr;
        vkDev.mapMemory(devParams, 0, paramSize, {} , &mem);
        memcpy(mem, parameters.get(), paramSize);

        vkDev.unmapMemory(devParams);

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
            //Camera
            {
                1,
                vk::DescriptorType::eUniformBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            },
            //Parameters
            {
                2,
                vk::DescriptorType::eUniformBuffer,
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

        vk::DescriptorBufferInfo outDescBufInfo  { outBuf,   0, texSize};
        vk::DescriptorBufferInfo camDescBufInfo  { camBuf,   0, camSize};
        vk::DescriptorBufferInfo paramDescBufInfo{ paramBuf, 0, paramSize};

        vk::WriteDescriptorSet outWriteDescSet{ vkDescSet
                                              , 0
                                              , 0
                                              , 1
                                              , vk::DescriptorType::eStorageBuffer
                                              , nullptr
                                              , &outDescBufInfo};

        vk::WriteDescriptorSet camWriteDescSet{ vkDescSet
                                              , 1
                                              , 0
                                              , 1
                                              , vk::DescriptorType::eUniformBuffer
                                              , nullptr
                                              , &camDescBufInfo};
        vk::WriteDescriptorSet paramsWriteDescSet { vkDescSet
                                                  , 2
                                                  , 0
                                                  , 1
                                                  , vk::DescriptorType::eUniformBuffer
                                                  , nullptr
                                                  , &paramDescBufInfo};

        vkDev.updateDescriptorSets(outWriteDescSet.descriptorCount,    &outWriteDescSet,     0, nullptr);
        vkDev.updateDescriptorSets(camWriteDescSet.descriptorCount,    &camWriteDescSet,    0, nullptr);
        vkDev.updateDescriptorSets(paramsWriteDescSet.descriptorCount, &paramsWriteDescSet, 0, nullptr);

        /**
         * Time to create the compute pipeline
         */

        uint32_t fileLength = 0;
        uint32_t* shaderCode = nullptr;

        if(shaderName == "")
        {
            throw(std::runtime_error("Shader name unset, something went quite wrong"));
        }

        shaderCode = readFile(fileLength, shaderName.c_str());

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
                                                       , "traceRegion"};

        vk::PipelineLayoutCreateInfo ppLayoutCreate{ {}
                                                   , 1
                                                   , &vkDescLayout};

        vkDev.createPipelineLayout(&ppLayoutCreate, nullptr, &vkPpLayout);

        vk::ComputePipelineCreateInfo pplCreate{ {} , shStageCreate, vkPpLayout};
        vkPpl = static_cast<vk::Pipeline&&>(vkDev.createComputePipeline({}, pplCreate));

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

        vkCmdBuffer.dispatch( static_cast<uint32_t>( ceil(cam->ScreenWidth()  / WORKGROUP_SIZE))
                            , static_cast<uint32_t>( ceil(cam->ScreenHeight() / WORKGROUP_SIZE))
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
        memcpy(outBuffer->data(),mem,texSize);
        vkDev.unmapMemory(devVec);

        /**
          * And clean up after ourselves. Even with the C++
          * interface, there's a lot of manual cleaning that
          * needs to be done
          */

        vkDev.free(vkCmdPool,1,&vkCmdBuffer, VULKAN_HPP_DEFAULT_DISPATCHER);
        vkDev.destroy(vkCmdPool, nullptr);
        vkDev.destroy(vkPpl, nullptr);
        vkDev.destroy(vkPpLayout, nullptr);
        vkDev.destroy(vkShMod, nullptr);
        vkDev.destroy(vkDescPool,nullptr);
        vkDev.destroy(vkDescLayout,nullptr);
        vkDev.free(devVec, nullptr);
        vkDev.free(devCam, nullptr);
        vkDev.free(devParams, nullptr);
        vkDev.destroy(outBuf, nullptr);
        vkDev.destroy(camBuf, nullptr);
        vkDev.destroy(paramBuf, nullptr);
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what();
    }

}

void FracGen::listDevices()
{
    try
    {
        vk::ApplicationInfo appInfo( "Toybrot Vulkan"
                                   , VK_MAKE_VERSION(1,0,0)
                                   , "No Engine"
                                   , VK_MAKE_VERSION(0,0,0)
                                   , VK_API_VERSION_1_2 );

        std::vector<const char*> enabledLayers;
        std::vector<const char*> enabledExtensions;

        if(enableValidation)
        {
            enableValidationLayer(enabledLayers, enabledExtensions);
        }

        vk::InstanceCreateInfo createInfo( vk::InstanceCreateFlags()
                                         , &appInfo
                                         , static_cast<uint>(enabledLayers.size())
                                         , enabledLayers.data()
                                         , static_cast<uint>(enabledExtensions.size())
                                         , enabledExtensions.data());


        vk::createInstance(&createInfo, nullptr, &vkInst);

        if(enabledLayers.size() > 0)
        {

            vkFuncLoad.init(vkInst, vkGetInstanceProcAddr);


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

        if(enableValidation)
        {
            std::vector<vk::LayerProperties> layerProperties{vk::enumerateInstanceLayerProperties()};

            std::cout << "Supported Vulkan layers: " << std::endl;
            for (vk::LayerProperties& prop : layerProperties)
            {
                std::cout << "    " << prop.layerName <<std::endl;
            }
        }

        std::vector<vk::PhysicalDevice> vkDevices{vkInst.enumeratePhysicalDevices()};

        if(vkDevices.size() > 0)
        {
            std::cout <<"Found "<< vkDevices.size() << " available Vulkan devices" << std::endl;

            vk::PhysicalDeviceProperties props;

            for(uint32_t i = 0; i < vkDevices.size(); i++)
            {
                vkDevices[i].getProperties(&props);
                std::vector< vk::QueueFamilyProperties > queueProps{vkDevices[i].getQueueFamilyProperties()};
                bool computeQueue = false;
                for( auto& q: queueProps)
                {
                    if( !(q.queueFlags & vk::QueueFlagBits::eGraphics)
                      && (q.queueFlags & vk::QueueFlagBits::eCompute ))
                    {
                        computeQueue = true;
                        break;
                    }
                }

                std::cout << "Device #"<< i <<std::endl;
                std::cout << "    Name: " << props.deviceName << std::endl;
                std::cout << "    Supported version:" << VK_VERSION_MAJOR(props.apiVersion) << "."
                                                      << VK_VERSION_MINOR(props.apiVersion) << "."
                                                      << VK_VERSION_PATCH(props.apiVersion)
                                                      << std::endl;

                std::cout << "    Exclusive compute queue " << (computeQueue?"":"not ") << "available" << std::endl << std::endl;

                if(enableValidation)
                {
                    std::vector<vk::ExtensionProperties> extensions{vkDevices[i].enumerateDeviceExtensionProperties()};

                    std::cout << std::endl << "Supported Vulkan extensions: " << std::endl;
                    for (vk::ExtensionProperties& ext : extensions)
                    {
                        std::cout << "    " << ext.extensionName <<std::endl;
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what();
    }
    if (enableValidation)
    {
        // destroy callback.
        vkInst.destroyDebugReportCallbackEXT(debugCallback, nullptr, vkFuncLoad);
    }
}

void FracGen::enableValidationLayer(std::vector<const char*>& enabledLayers, std::vector<const char*>& enabledExtensions)
{
    std::vector<vk::LayerProperties> layerProperties{vk::enumerateInstanceLayerProperties()};


    // And then we simply check if VK_LAYER_KHRONOS_validation is among the supported layers.

    bool foundLayer = false;
    for (vk::LayerProperties prop : layerProperties)
    {
        //std::cout << "Found layer: " << prop.layerName <<std::endl;
        if (strcmp("VK_LAYER_KHRONOS_validation", prop.layerName) == 0)
        {
            foundLayer = true;
            break;
        }
    }

    if (!foundLayer)
    {
        throw std::runtime_error("Layer VK_LAYER_KHRONOS_validation not supported\n");
    }
    enabledLayers.push_back("VK_LAYER_KHRONOS_validation");

    std::vector<vk::ExtensionProperties> extensions{vk::enumerateInstanceExtensionProperties()};

    bool foundReportExtension = false;
    if(extensions.size() > 0)
    {
        for(const auto& ext : extensions)
        {
            //std::cout << ext.extensionName <<std::endl;
            if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, ext.extensionName) == 0)
            {
                foundReportExtension = true;
                continue;
            }
        }
    }

    if (!foundReportExtension)
    {
        std::string err ("Extension ");
        err.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        err.append(" not supported\n");
        throw std::runtime_error(err.c_str());
    }
    else
    {
        enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

}

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p, shaderSource backend, int requestedDevice)
    : bench{benching}
    , cam{c}
    , parameters{p}
{
    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());
    switch(backend)
    {
        case shaderSource::GLSL:
            shaderName = "FracGen.glsl";
            break;
        case shaderSource::HLSL:
            shaderName = "FracGen.hlsl";
            break;
        case shaderSource::OPENCL:
            shaderName = "FracGen.cl";
            break;
    }

#ifdef TOYBROT_USE_DOUBLES
    shaderName += ".db.spv";
#else
    shaderName += ".spv";
#endif

    static bool once = false;
    try
    {
        vk::ApplicationInfo appInfo( "Toybrot Vulkan"
                                   , VK_MAKE_VERSION(1,0,0)
                                   , "No Engine"
                                   , VK_MAKE_VERSION(0,0,0)
                                   , VK_API_VERSION_1_2 );


        // Check for validation layer
        std::vector<const char*> enabledLayers;
        std::vector<const char*> instanceExtensions;

        if(enableValidation)
        {
            enableValidationLayer(enabledLayers, instanceExtensions);
        }

        //Time to create the instance

        vk::InstanceCreateInfo createInfo( vk::InstanceCreateFlags()
                                         , &appInfo
                                         , static_cast<uint>(enabledLayers.size())
                                         , enabledLayers.data()
                                         , static_cast<uint>(instanceExtensions.size())
                                         , instanceExtensions.data());


        vk::createInstance(&createInfo, nullptr, &vkInst);


        if(enabledLayers.size() > 0)
        {

            vkFuncLoad.init(vkInst, vkGetInstanceProcAddr);


            vk::DebugReportFlagsEXT dbgFlags( vk::DebugReportFlagBitsEXT::eError
                                            | vk::DebugReportFlagBitsEXT::eInformation
                                            //| vk::DebugReportFlagBitsEXT::eDebug
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

//            for(uint32_t i = 0; i < vkDevices.size(); i++)
//            {
//                vkDevices[i].getProperties(&props);
//                if(!bench || !once)
//                {
//                    std::cout << "Device "<< i << " Name: " << props.deviceName << std::endl;
//                    std::cout << "Supported version:" << VK_VERSION_MAJOR(props.apiVersion) << "."
//                                                      << VK_VERSION_MINOR(props.apiVersion) << "."
//                                                      << VK_VERSION_PATCH(props.apiVersion)
//                                                      << std::endl;
//                }
//            }

            if(requestedDevice < 0)
            {
                for(uint32_t i = 0; (i < vkDevices.size()) && (!foundDevice); i++)
                {
                    vkDevices[i].getProperties(&props);

                    std::vector< vk::QueueFamilyProperties > queueProps{vkDevices[i].getQueueFamilyProperties()};

                    //Try to find an exclusive compute queue for the device;
                    for(uint32_t j = 0; j < queueProps.size(); j++)
                    {
                        if (  !(queueProps[j].queueFlags & vk::QueueFlagBits::eGraphics)
                            && (queueProps[j].queueFlags & vk::QueueFlagBits::eCompute )  )
                        {
                          deviceIndex = requestedDevice;
                          vkQIdx = j;
                          foundDevice = true;
                          vkPhys = vkDevices[requestedDevice];
                          exclusiveCompute = true;
                          break;
                        }
                    }
                    if(!foundDevice)
                    {
                        for(uint32_t j = 0; j < queueProps.size(); j++)
                        {
                            if ((queueProps[j].queueFlags & vk::QueueFlagBits::eCompute )  )
                            {
                              deviceIndex = requestedDevice;
                              vkQIdx = j;
                              foundDevice = true;
                              vkPhys = vkDevices[requestedDevice];
                              break;
                            }
                        }
                    }
                }
            }
            else
            {
                if(requestedDevice >= static_cast<int>(vkDevices.size()))
                {
                    throw std::runtime_error("Requested invalid device: Found " + std::to_string(vkDevices.size())
                                                                    + " Requested " + std::to_string(requestedDevice));
                }
                vkDevices[requestedDevice].getProperties(&props);

                std::vector< vk::QueueFamilyProperties > queueProps{vkDevices[requestedDevice].getQueueFamilyProperties()};

                //Try to find an exclusive compute queue for the device;
                for(uint32_t j = 0; j < queueProps.size(); j++)
                {
                    if (  !(queueProps[j].queueFlags & vk::QueueFlagBits::eGraphics)
                        && (queueProps[j].queueFlags & vk::QueueFlagBits::eCompute )  )
                    {
                      deviceIndex = requestedDevice;
                      vkQIdx = j;
                      foundDevice = true;
                      vkPhys = vkDevices[requestedDevice];
                      exclusiveCompute = true;
                      break;
                    }
                }
                if(!foundDevice)
                {
                    for(uint32_t j = 0; j < queueProps.size(); j++)
                    {
                        if ((queueProps[j].queueFlags & vk::QueueFlagBits::eCompute )  )
                        {
                          deviceIndex = requestedDevice;
                          vkQIdx = j;
                          foundDevice = true;
                          vkPhys = vkDevices[requestedDevice];
                          break;
                        }
                    }
                }

            }
            if((foundDevice) && (!bench || !once) )
            {
                once = true;

                std::cout << std::endl << "ToyBrot running Vulkan using shader source: ";
                switch (backend)
                {
                    case shaderSource::GLSL:
                        std::cout << "GLSL (" << shaderName << ")" << std::endl;
                        std::cout << "Compiled using glslangValidator" << std::endl << std::endl;
                        break;
                    case shaderSource::HLSL:
                        std::cout << "HLSL (" << shaderName << ")" << std::endl;
                        std::cout << "Compiled using glslangValidator" << std::endl << std::endl;
                        break;
                    case shaderSource::OPENCL:
                        std::cout << "OpenCL (" << shaderName << ")" << std::endl;
                        std::cout << "Compiled using clspv" << std::endl << std::endl;
                        break;
                }

                std::cout << vkDevices.size() << " available Vulkan devices" << std::endl << std::endl;

                std::cout << "Using "<< (requestedDevice < 0 ? "" : "requested ") << "device " << deviceIndex << ":" << std::endl;
                std::cout << "Name: "       << props.deviceName << std::endl;
                std::cout << "Using a " << (exclusiveCompute? "dedicated":"shared") << " compute queue" << std::endl;
                std::cout << "DeviceID: "   << props.deviceID << std::endl;
                std::cout << "VendorID: "   << props.vendorID << std::endl;
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
#ifdef TOYBROT_USE_DOUBLES
        devFeatures.setShaderFloat64(true);
#endif

        vk::DeviceCreateInfo deviceInfo{ {} , 1 , &queueInfo };
        deviceInfo.setPEnabledFeatures(&devFeatures);

        std::vector<const char*> deviceExtensions;

        if(enableValidation)
        {
            //Enable non_semantic_info extension if present for printfEXT in shader
            std::vector<vk::ExtensionProperties> extensions{vkDevices[deviceIndex].enumerateDeviceExtensionProperties()};

            bool foundNonSemanticInfoExtension = false;
            if(extensions.size() > 0)
            {
                for(const auto& ext : extensions)
                {

                    if (strcmp(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, ext.extensionName) == 0)
                    {
                        foundNonSemanticInfoExtension = true;
                        continue;
                    }
                }
            }

            if (!foundNonSemanticInfoExtension)
            {
                std::string err ("Extension ");
                err.append(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
                err.append(" not supported by device\n");
                //throw std::runtime_error(err.c_str());
                std::cout << err;
            }
            else
            {
                deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
            }
        }
        deviceInfo.setEnabledExtensionCount(deviceExtensions.size());
        deviceInfo.setPpEnabledExtensionNames(deviceExtensions.data());

        vkPhys.createDevice(&deviceInfo, nullptr, &vkDev);
        vkDev.getQueue(vkQIdx, 0, &vkQ);
        vkPhys.getMemoryProperties(&vkMProps);
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        exit(1);
    }
}

FracGen::~FracGen()
{
    if (enableValidation)
    {
        // destroy callback.
        vkInst.destroyDebugReportCallbackEXT(debugCallback, nullptr, vkFuncLoad);
    }

    //vkDev.destroy();
    //vkInst.destroy();
}
