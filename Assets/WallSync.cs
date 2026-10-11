using UnityEngine;

public class WallSync : MonoBehaviour
{
    [SerializeField] private float cameraSize = 50f;

    private GameObject[] walls;
    private GameObject divider;
    private Camera main_camera;

    private void OnValidate()
    {
        main_camera = Camera.main;
        main_camera.orthographicSize = cameraSize;

        divider = GameObject.Find("Divider");
        walls = GameObject.FindGameObjectsWithTag("Wall");

        walls[3].transform.localScale = new Vector3(10, cameraSize * 2f);
        walls[3].transform.position = new Vector3((-cameraSize * 16f) / 9f - 5f, 0);

        walls[2].transform.localScale = new Vector3(10, cameraSize * 2f);
        walls[2].transform.position = new Vector3((cameraSize * 16f) / 9f + 5f, 0);

        walls[1].transform.localScale = new Vector3(10, walls[2].transform.position.x * 2 + 10);
        walls[1].transform.position = new Vector3(0, cameraSize + 5);

        walls[0].transform.localScale = new Vector3(10, walls[2].transform.position.x * 2 + 10);
        walls[0].transform.position = new Vector3(0, -cameraSize - 5);

        divider.transform.localScale = new Vector3(1, walls[0].transform.localScale.y - 20);
    }
}