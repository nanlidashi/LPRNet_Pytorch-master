package com.example.mpdemo.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.mpdemo.entity.Order;
import com.example.mpdemo.entity.User;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface OrderMapper extends BaseMapper<Order> {

    @Select("select * from t_order where uid = #{uid}")
    List<Order> selectByUid(int uid);


    @Select("select * from t_order")
    @Results({
            @Result(column = "id", property = "id"),
            @Result(column = "total", property = "total"),
            @Result(column = "ordertime", property = "ordertime"),
            @Result(column = "uid", property = "user",javaType = User.class,
                    one = @One(select = "com.example.mpdemo.mapper.UserMapper.selectById"))
    })
    List<Order> selectAllOrderAndUser();

}
