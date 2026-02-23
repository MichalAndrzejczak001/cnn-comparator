package spring.project.thesis.logicbackend;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class TestController {
    @GetMapping("/get1")
    public String get1() {
        return "get1";
    }

    @GetMapping("/get2")
    public String get2() {
        return "get2";
    }

    @GetMapping("/get3")
    public String get3() {
        return "get3";
    }
}
