package org.example.helloworld.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import springfox.documentation.builders.ApiInfoBuilder;
import springfox.documentation.builders.PathSelectors;
import springfox.documentation.builders.RequestHandlerSelectors;
import springfox.documentation.oas.annotations.EnableOpenApi;
import springfox.documentation.service.ApiInfo;
import springfox.documentation.spi.DocumentationType;
import springfox.documentation.spring.web.plugins.Docket;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

@Configuration //告诉spring boot 这是一个配置类
@EnableOpenApi
public class SwaggerConfig {
    /**
     * 配置swagger2相关的bean
     */
    @Bean
    public Docket createRestApi() {//页面为https://localhost:8081/swagger-ui/index.html
        return new Docket(DocumentationType.SWAGGER_2)
                // 配置文档类型
                .apiInfo(apiInfo()) //配置文档信息
                .select()
                // 扫描指定包下的所有接口
                .apis(RequestHandlerSelectors.basePackage("org"))
                // 过滤指定路径下的接口
                .paths(PathSelectors.any())
                .build();
    }
    /*
     * 配置文档信息
     */
    private ApiInfo apiInfo() {
        return new ApiInfoBuilder()
                .title("演示项目 APIs") //标题
                .description("演示项目") //演示项目
                .version("1.0") //版本
                .build();
    }
}
